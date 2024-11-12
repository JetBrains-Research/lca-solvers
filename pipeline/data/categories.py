from typing import Literal

CategoryType = Literal['InCommit', 'InFile', 'InProject', 'NonInformative', 'Other', 'OtherAPI', 'TODO']

ID2CATEGORY = [
    'InCommit',
    'InFile',
    'InProject',
    'NonInformative',
    'Other',
    'OtherAPI',
    'TODO',
]
CATEGORY2ID = {category: i for i, category in enumerate(ID2CATEGORY)}
UNDEFINED_CATEGORY_ID = -1
