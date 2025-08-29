import tkinter as tk
from tkinter import Text, Scrollbar, Button, StringVar, OptionMenu, Checkbutton, filedialog, Toplevel, Label, Entry
import ollama
import numpy as np
import networkx as nx
import pickle
import os
import logging

logging.basicConfig(level=logging.DEBUG)

class EnglishLearningChatbot:
    def __init__(self):
        self.conversation_history = []
        self.model = 'deepseek-r1'
        self.embedding_model = 'nomic-embed-text'
        self.models = {'deepseek-r1': 'deepseek-r1'}
        self.system_role = None
        self.ollama_params = {
            'mirostat': 0,
            'mirostat_eta': 0.1,
            'mirostat_tau': 5.0,
            'num_ctx': 8000,
            'num_gpu': 49,
            'num_thread': 24,
            'repeat_last_n': 64,
            'repeat_penalty': 1.2,
            'temperature': 0.8,
            'seed': 0,
            'stop': None,
            'tfs_z': 1.0,
            'top_k': 50,
            'top_p': 0.9
        }
        self.ollama_params_enabled = {
            'mirostat': False,
            'mirostat_eta': False,
            'mirostat_tau': False,
            'num_ctx': False,
            'num_gpu': False,
            'num_thread': False,
            'repeat_last_n': False,
            'repeat_penalty': False,
            'temperature': True,
            'seed': False,
            'stop': False,
            'tfs_z': False,
            'top_k': True,
            'top_p': True
        }
        self.long_term_memory = []
        self.embeddings = []
        self.importances = []
        self.knowledge_graph = nx.DiGraph()  # Graphe dirigé
        self.correlation_threshold = 0.7
        self.decay_rate = 0.95
        self.boost = 0.1
        self.min_importance = 0.1
        self.max_memory_size = 1000
        self.memory_file = 'long_term_memory.pkl'
        self.embeddings_file = 'embeddings.pkl'
        self.importances_file = 'importances.pkl'
        self.graph_file = 'knowledge_graph.pkl'
        self.load_long_term_memory()

    def set_model(self, model):
        self.model = model

    def set_system_role(self, role):
        self.system_role = role if role else None

    def set_ollama_params(self, params, enabled):
        self.ollama_params = params
        self.ollama_params_enabled = enabled

    def get_response(self, user_input):
        relevant_knowledge = self.retrieve_relevant_knowledge(user_input)
        conversation = self.conversation_history.copy()
        if relevant_knowledge:
            conversation.append({'role': 'system', 'content': f"Connaissances pertinentes : {relevant_knowledge}"})
        if self.system_role:
            conversation.append({'role': 'system', 'content': self.system_role})
        conversation.append({'role': 'user', 'content': user_input})

        filtered_params = {k: v for k, v in self.ollama_params.items() if self.ollama_params_enabled[k] and v is not None}
        response = ollama.chat(model=self.model, messages=conversation, options=filtered_params)
        response_content = response['message']['content'].strip()
        conversation.append({'role': 'assistant', 'content': response_content})

        self.conversation_history = conversation

        extract_prompt = (
            "Extrait les déclarations factuelles clés de la conversation suivante. "
            "Ignore les opinions, les redondances, et les informations non factuelles. "
            "Liste-les une par ligne sans numéros."
        )
        extract_messages = [
            {'role': 'system', 'content': extract_prompt},
            {'role': 'user', 'content': user_input + "\nAssistant: " + response_content}
        ]
        extract_response = ollama.chat(model=self.model, messages=extract_messages, options=filtered_params)
        new_facts = [f.strip() for f in extract_response['message']['content'].split("\n") if f.strip()]

        if relevant_knowledge and new_facts:
            synth_prompt = (
                "Basé sur les faits suivants et la conversation actuelle, propose une ou deux généralisations ou inférences "
                "qui relient ces informations à un concept plus large ou à un domaine différent."
                "Inclut des analogies ou des liens avec des domaines non mentionnés dans la conversation."
                f"Faits : {relevant_knowledge}\nConversation : {user_input}\nRéponse : {response_content}"
            )
            synth_messages = [{'role': 'system', 'content': synth_prompt}]
            synth_response = ollama.chat(model=self.model, messages=synth_messages, options=filtered_params)
            synthesized_facts = [f.strip() for f in synth_response['message']['content'].split("\n") if f.strip()]
            new_facts.extend(synthesized_facts)

        self.add_to_long_term_memory(new_facts)
        self.update_graph()
        self.prune_memory()
        self.save_long_term_memory()

        return response_content, new_facts

    def add_to_long_term_memory(self, new_facts):
        for fact in new_facts:
            embedding = np.array(ollama.embeddings(model=self.embedding_model, prompt=fact)['embedding'])
            similar = False
            for i, emb in enumerate(self.embeddings):
                sim = np.dot(embedding, emb) / (np.linalg.norm(embedding) * np.linalg.norm(emb) + 1e-8)
                if sim > 0.9:
                    similar = True
                    self.importances[i] = min(1.0, self.importances[i] + self.boost)
                    break
            if not similar:
                fact_id = len(self.long_term_memory)
                self.long_term_memory.append(fact)
                self.embeddings.append(embedding)
                self.importances.append(1.0)
                self.knowledge_graph.add_node(fact_id, fact=fact, importance=1.0)
                for i in range(fact_id):
                    emb_i = self.embeddings[i]
                    sim = np.dot(embedding, emb_i) / (np.linalg.norm(embedding) * np.linalg.norm(emb_i) + 1e-8)
                    if sim > self.correlation_threshold:
                        if self.is_more_general(fact, self.long_term_memory[i]):
                            self.knowledge_graph.add_edge(i, fact_id, weight=sim)
                        else:
                            self.knowledge_graph.add_edge(fact_id, i, weight=sim)

    def is_more_general(self, fact1, fact2):
        prompt = (
            f"Entre ces deux faits, lequel est le plus général (couvre un concept plus large) ?\n"
            f"Fait 1 : {fact1}\nFait 2 : {fact2}\n"
            "Réponds uniquement avec 'Fait 1' ou 'Fait 2'."
        )
        response = ollama.chat(model=self.model, messages=[{'role': 'system', 'content': prompt}])
        return response['message']['content'].strip() == 'Fait 1'

    def update_graph(self):
        for i in range(len(self.embeddings)):
            for j in range(i + 1, len(self.embeddings)):
                sim = np.dot(self.embeddings[i], self.embeddings[j]) / (
                    np.linalg.norm(self.embeddings[i]) * np.linalg.norm(self.embeddings[j]) + 1e-8
                )
                if sim > self.correlation_threshold:
                    if self.is_more_general(self.long_term_memory[i], self.long_term_memory[j]):
                        self.knowledge_graph.add_edge(i, j, weight=sim)
                    else:
                        self.knowledge_graph.add_edge(j, i, weight=sim)
                elif (i, j) in self.knowledge_graph.edges or (j, i) in self.knowledge_graph.edges:
                    self.knowledge_graph.remove_edge(i, j)
                    self.knowledge_graph.remove_edge(j, i)

    def prune_memory(self):
        if len(self.long_term_memory) <= self.max_memory_size:
            return
        indices_to_remove = [i for i, imp in enumerate(self.importances) if imp < self.min_importance]
        for idx in sorted(indices_to_remove, reverse=True):
            self.knowledge_graph.remove_node(idx)
            del self.long_term_memory[idx]
            del self.embeddings[idx]
            del self.importances[idx]
        mapping = {old: new for new, old in enumerate(sorted(self.knowledge_graph.nodes))}
        self.knowledge_graph = nx.relabel_nodes(self.knowledge_graph, mapping)

    def retrieve_relevant_knowledge(self, query):
        if not self.embeddings:
            return ""
        query_emb = np.array(ollama.embeddings(model=self.embedding_model, prompt=query)['embedding'])
        sims = [np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-8) for emb in self.embeddings]
        top_indices = np.argsort(sims)[-5:][::-1]
        relevant_ids = set(top_indices)
        for idx in list(relevant_ids):
            for neigh in self.knowledge_graph.successors(idx):  # Utilise successors pour DiGraph
                if self.knowledge_graph.edges[idx, neigh]['weight'] > 0.5:
                    relevant_ids.add(neigh)
        relevant_facts = []
        used_ids = set()
        for i in relevant_ids:
            if self.importances[i] > self.min_importance:
                relevant_facts.append(self.long_term_memory[i])
                used_ids.add(i)
        for i in range(len(self.importances)):
            if i in used_ids:
                self.importances[i] = min(1.0, self.importances[i] + self.boost)
            else:
                self.importances[i] *= self.decay_rate
        return "\n".join(relevant_facts)

    def save_long_term_memory(self):
        with open(self.memory_file, 'wb') as f:
            pickle.dump(self.long_term_memory, f)
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(self.embeddings, f)
        with open(self.importances_file, 'wb') as f:
            pickle.dump(self.importances, f)
        with open(self.graph_file, 'wb') as f:
            pickle.dump(self.knowledge_graph, f)

    def load_long_term_memory(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'rb') as f:
                self.long_term_memory = pickle.load(f)
        if os.path.exists(self.embeddings_file):
            with open(self.embeddings_file, 'rb') as f:
                self.embeddings = pickle.load(f)
        if os.path.exists(self.importances_file):
            with open(self.importances_file, 'rb') as f:
                self.importances = pickle.load(f)
        if os.path.exists(self.graph_file):
            with open(self.graph_file, 'rb') as f:
                graph = pickle.load(f)
                if isinstance(graph, nx.Graph) and not isinstance(graph, nx.DiGraph):
                    # Convertir Graph en DiGraph
                    self.knowledge_graph = nx.DiGraph(graph)
                    logging.info("Converted loaded Graph to DiGraph")
                else:
                    self.knowledge_graph = graph
        else:
            self.knowledge_graph = nx.DiGraph()  # Assurer DiGraph si aucun fichier

    def clear_history(self):
        self.conversation_history = []

class EnglishLearningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chatbot")
        self.chatbot = EnglishLearningChatbot()
        self.setup_gui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_gui(self):
        frame = tk.Frame(self.root, padx=10, pady=10)
        frame.pack(padx=10, pady=10, expand=True, fill='both')

        model_label = tk.Label(frame, text="Select a model:")
        model_label.pack()

        self.model_var = StringVar(self.root)
        self.model_var.set('deepseek-r1')
        model_options = list(self.chatbot.models.keys())
        model_menu = OptionMenu(frame, self.model_var, *model_options, command=self.change_model)
        model_menu.pack()

        role_label = tk.Label(frame, text="Import system role from file:")
        role_label.pack()

        self.role_entry = tk.Entry(frame, width=50)
        self.role_entry.pack()

        select_role_btn = Button(frame, text="Select Role File", command=self.select_role_file)
        select_role_btn.pack()

        self.use_role_var = tk.BooleanVar()
        use_role_checkbox = Checkbutton(frame, text="Use System Role", variable=self.use_role_var)
        use_role_checkbox.pack()

        clear_history_btn = Button(frame, text="Clear History", command=self.clear_history)
        clear_history_btn.pack()

        parameters_btn = Button(frame, text="Parameters & Options", command=self.open_parameters_window)
        parameters_btn.pack()

        validate_facts_btn = Button(frame, text="Validate Facts", command=self.validate_facts)
        validate_facts_btn.pack()

        entry_label = tk.Label(frame, text="Enter your message:")
        entry_label.pack()

        entry_frame = tk.Frame(frame)
        entry_frame.pack(fill='both', expand=True)

        entry_scrollbar = Scrollbar(entry_frame)
        entry_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.entry = Text(entry_frame, height=5, width=50, yscrollcommand=entry_scrollbar.set)
        self.entry.pack(fill='both', expand=True)
        entry_scrollbar.config(command=self.entry.yview)

        submit_btn = Button(frame, text="Submit", command=self.submit)
        submit_btn.pack()

        result_label = tk.Label(frame, text="Chatbot response:")
        result_label.pack()

        result_frame = tk.Frame(frame)
        result_frame.pack(fill='both', expand=True)

        result_scrollbar = Scrollbar(result_frame)
        result_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.result = Text(result_frame, height=10, width=50, yscrollcommand=result_scrollbar.set)
        self.result.pack(fill='both', expand=True)
        result_scrollbar.config(command=self.result.yview)

        self.facts_to_validate = []

    def change_model(self, model):
        self.chatbot.set_model(model)

    def select_role_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            self.role_entry.delete(0, tk.END)
            self.role_entry.insert(0, file_path)
            try:
                with open(file_path, 'r') as file:
                    role_content = file.read().strip()
                    self.chatbot.set_system_role(role_content)
                    print(f"System role imported: {role_content}")
            except FileNotFoundError:
                print(f"File not found: {file_path}")

    def clear_history(self):
        self.chatbot.clear_history()
        print("Conversation history cleared.")

    def open_parameters_window(self):
        parameters_window = Toplevel(self.root)
        parameters_window.title("Parameters & Options")

        Label(parameters_window, text="Parameters & Options").grid(row=0, column=0, columnspan=3, pady=10)

        params = [
            ('mirostat', StringVar(value=str(self.chatbot.ollama_params['mirostat'])), tk.BooleanVar(value=self.chatbot.ollama_params_enabled['mirostat'])),
            ('mirostat_eta', StringVar(value=str(self.chatbot.ollama_params['mirostat_eta'])), tk.BooleanVar(value=self.chatbot.ollama_params_enabled['mirostat_eta'])),
            ('mirostat_tau', StringVar(value=str(self.chatbot.ollama_params['mirostat_tau'])), tk.BooleanVar(value=self.chatbot.ollama_params_enabled['mirostat_tau'])),
            ('num_ctx', StringVar(value=str(self.chatbot.ollama_params['num_ctx'])), tk.BooleanVar(value=self.chatbot.ollama_params_enabled['num_ctx'])),
            ('num_gpu', StringVar(value=str(self.chatbot.ollama_params['num_gpu'])), tk.BooleanVar(value=self.chatbot.ollama_params_enabled['num_gpu'])),
            ('num_thread', StringVar(value=str(self.chatbot.ollama_params['num_thread'])), tk.BooleanVar(value=self.chatbot.ollama_params_enabled['num_thread'])),
            ('repeat_last_n', StringVar(value=str(self.chatbot.ollama_params['repeat_last_n'])), tk.BooleanVar(value=self.chatbot.ollama_params_enabled['repeat_last_n'])),
            ('repeat_penalty', StringVar(value=str(self.chatbot.ollama_params['repeat_penalty'])), tk.BooleanVar(value=self.chatbot.ollama_params_enabled['repeat_penalty'])),
            ('temperature', StringVar(value=str(self.chatbot.ollama_params['temperature'])), tk.BooleanVar(value=self.chatbot.ollama_params_enabled['temperature'])),
            ('seed', StringVar(value=str(self.chatbot.ollama_params['seed'])), tk.BooleanVar(value=self.chatbot.ollama_params_enabled['seed'])),
            ('stop', StringVar(value=self.chatbot.ollama_params['stop'] or ''), tk.BooleanVar(value=self.chatbot.ollama_params_enabled['stop'])),
            ('tfs_z', StringVar(value=str(self.chatbot.ollama_params['tfs_z'])), tk.BooleanVar(value=self.chatbot.ollama_params_enabled['tfs_z'])),
            ('top_k', StringVar(value=str(self.chatbot.ollama_params['top_k'])), tk.BooleanVar(value=self.chatbot.ollama_params_enabled['top_k'])),
            ('top_p', StringVar(value=str(self.chatbot.ollama_params['top_p'])), tk.BooleanVar(value=self.chatbot.ollama_params_enabled['top_p']))
        ]

        for i, (param_name, param_var, param_enabled_var) in enumerate(params):
            row = i + 1
            Label(parameters_window, text=param_name.capitalize()).grid(row=row, column=0, padx=10, pady=5, sticky='w')
            Entry(parameters_window, textvariable=param_var).grid(row=row, column=1, padx=10, pady=5)
            Checkbutton(parameters_window, text="Enable", variable=param_enabled_var).grid(row=row, column=2, padx=10, pady=5)

            setattr(self, f"{param_name}_var", param_var)
            setattr(self, f"{param_name}_enabled_var", param_enabled_var)

        save_btn = Button(parameters_window, text="Save", command=self.save_parameters)
        save_btn.grid(row=len(params) + 1, column=0, columnspan=3, pady=10)

    def save_parameters(self):
        int_params = ['mirostat', 'num_ctx', 'num_gpu', 'num_thread', 'repeat_last_n', 'seed', 'top_k']
        float_params = ['mirostat_eta', 'mirostat_tau', 'repeat_penalty', 'temperature', 'tfs_z', 'top_p']

        params = {}
        for param in self.chatbot.ollama_params.keys():
            var = getattr(self, f"{param}_var")
            value_str = var.get()
            try:
                if param in int_params:
                    params[param] = int(value_str) if value_str else 0
                elif param in float_params:
                    params[param] = float(value_str) if value_str else 0.0
                elif param == 'stop':
                    params[param] = value_str if value_str else None
            except ValueError:
                print(f"Invalid value for {param}: {value_str}")
                return

        enabled = {
            'mirostat': self.mirostat_enabled_var.get(),
            'mirostat_eta': self.mirostat_eta_enabled_var.get(),
            'mirostat_tau': self.mirostat_tau_enabled_var.get(),
            'num_ctx': self.num_ctx_enabled_var.get(),
            'num_gpu': self.num_gpu_enabled_var.get(),
            'num_thread': self.num_thread_enabled_var.get(),
            'repeat_last_n': self.repeat_last_n_enabled_var.get(),
            'repeat_penalty': self.repeat_penalty_enabled_var.get(),
            'temperature': self.temperature_enabled_var.get(),
            'seed': self.seed_enabled_var.get(),
            'stop': self.stop_enabled_var.get(),
            'tfs_z': self.tfs_z_enabled_var.get(),
            'top_k': self.top_k_enabled_var.get(),
            'top_p': self.top_p_enabled_var.get()
        }
        self.chatbot.set_ollama_params(params, enabled)
        print("Parameters saved:", params)
        print("Parameters enabled:", enabled)

    def validate_facts(self):
        if not self.facts_to_validate:
            print("No facts to validate.")
            return

        validate_window = Toplevel(self.root)
        validate_window.title("Validate Facts")

        Label(validate_window, text="Validez ou rejetez les faits extraits :").pack(pady=5)

        self.fact_vars = []
        for fact in self.facts_to_validate:
            var = tk.BooleanVar(value=True)
            Checkbutton(validate_window, text=fact, variable=var).pack(anchor='w', padx=10)
            self.fact_vars.append((fact, var))

        Button(validate_window, text="Confirm", command=self.confirm_facts).pack(pady=10)

    def confirm_facts(self):
        validated_facts = [fact for fact, var in self.fact_vars if var.get()]
        self.chatbot.add_to_long_term_memory(validated_facts)
        self.chatbot.update_graph()
        self.chatbot.prune_memory()
        self.chatbot.save_long_term_memory()
        self.facts_to_validate = []
        print("Validated facts added to memory:", validated_facts)

    def submit(self):
        user_input = self.entry.get("1.0", "end-1c")
        if self.use_role_var.get() and self.chatbot.system_role:
            response, new_facts = self.chatbot.get_response(user_input)
        else:
            self.chatbot.system_role = None
            response, new_facts = self.chatbot.get_response(user_input)
        self.result.delete("1.0", tk.END)
        self.result.insert(tk.END, response)
        self.facts_to_validate = new_facts
        print("Facts to validate:", new_facts)

    def on_closing(self):
        self.chatbot.save_long_term_memory()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = EnglishLearningApp(root)
    root.mainloop()
