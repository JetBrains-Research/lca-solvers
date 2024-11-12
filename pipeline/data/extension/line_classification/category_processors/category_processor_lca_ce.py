from pipeline.data.extension.line_classification.category_processors.category_processor_base import CategoryProcessorBase


class CategoryProcessorCodeEngine(CategoryProcessorBase):
    def choose_main_category(self, line_categories: list[str]) -> str:
        if 'NonInformative' in line_categories and len(line_categories) > 1:
            return 'TODO'
        elif 'InCommit' in line_categories:
            return 'InCommit'
        elif 'InProject' in line_categories:
            return 'InProject'
        elif 'InFile' in line_categories:
            return 'InFile'
        elif 'OtherAPI' in line_categories:
            return 'OtherAPI'
        elif 'NonInformative' in line_categories:
            return 'NonInformative'
        elif line_categories == ['Other']:
            return 'Other'
        else:
            return 'NoCategory'
