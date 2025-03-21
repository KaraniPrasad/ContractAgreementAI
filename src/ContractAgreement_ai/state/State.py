class UserState:
    def __init__(self):
        self.user_type = None  # Buyer or Supplier
        self.clauses = []
        self.context = []
        self.previous_interactions = []

    def set_user_type(self, user_type):
        self.user_type = user_type

    def add_clause(self, clause):
        self.clauses.append(clause)

    def add_to_context(self, context):
        self.context.append(context)

    def add_interaction(self, interaction):
        self.previous_interactions.append(interaction)
