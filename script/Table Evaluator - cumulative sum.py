from table_evaluator import load_data, TableEvaluator

real, fake = load_data('NSL-KDD/KDDTrain+.csv', 'NSL-KDD/synthetic_data.csv')

cat_cols = ["'protocol_type'", "'service'", "'flag'", "'land'", "'logged_in'", "'is_host_login'", "'is_guest_login'", "'class'"]

table_evaluator = TableEvaluator(real, fake, cat_cols=cat_cols)

table_evaluator.visual_evaluation()
