from datasets import Features, Value, Sequence

completion_file_features_local = Features({
    'repo': Value('string'),
    'commit_hash': Value('string'),
    'snapshot_hash': Value('string'),
    'committer_date': Value('string'),
    'year': Value('int32'),
    'datapoint_identifier': Value('string'),
    'filename': Value('string'),
    'content': Value('string'),
    'total_lines': Value('int32'),
    'total_chars': Value('int32'),
    'lines': Sequence({
        'line_idx': Value('int32'),
        'line_num': Value('int32'),
        'categories': Sequence(Value('string')),
        'category_processor_name': Value('string'),
        'main_category': Value('string'),
    }),
    'repo_num_chars_relevant': Value('int32'),
    'repo_num_chars_total': Value('int32'),
    'repo_num_chars_name': Value('string')
})
