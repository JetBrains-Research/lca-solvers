from pipeline.data.extension.line_classification.category_processors.category_processor_base import CategoryProcessorBase


class CategoryProcessorDummy(CategoryProcessorBase):
    def choose_main_category(self, line_categories: list[str]) -> str:
        if len(line_categories) == 0:
            return 'NoCategory'
        return line_categories[0]
