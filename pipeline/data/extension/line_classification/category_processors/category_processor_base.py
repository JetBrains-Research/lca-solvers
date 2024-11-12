class CategoryProcessorBase:
    def __init__(self):
        pass
    def choose_main_category(self, line_categories: list[str]) -> str:
        raise NotImplementedError
