from datasets import Features, Sequence, Value

# snapshot_features = Features({
#     'files': Sequence({
#         'filename': Value('string'),
#         'content': Value('string'),
#     }),
#     'year': Value('int32'),
#     'repo': Value('string'),
#     'commit_hash': Value('string'),
#     'completion_file_commit_hash': Value('string'),
#     'relevant_extensions': Sequence(Value('string')),  # list of strings for file extensions
#     'num_chars_relevant': Value('int32'),
#     'num_chars_total': Value('int32'),
# })

snapshot_features = Features({
    'filename': Value('string'),
    'content': Value('string'),
    'completion_file_commit_hash': Value('string'),
    'repo_snapshot_commit_hash': Value('string'),
})
