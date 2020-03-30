########### Define Output, Covariate and Treatment columns names for postgreSQL Dataset ########
target_columns    = ['local_written_blocks', 'temp_written_blocks', 'shared_hit_blocks', 'temp_read_blocks', 'local_read_blocks', 'runtime', 'shared_read_blocks']
treatment_columns = ['index_level', 'page_cost', 'memory_level']
covariate_columns = ['rows', 'creation_year', 'num_ref_tables', 'num_joins', 'num_group_by', 'queries_by_user', 'length_chars', 'total_ref_rows', 'local_hit_blocks', 'favorite_count']
feature_columns   = covariate_columns.copy()
feature_columns.extend(treatment_columns)


