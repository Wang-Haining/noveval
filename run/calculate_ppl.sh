python3 calculate_ppl_acl.py \
--out_dir out_wikipedia_en \
--device cuda:0 \
--ppl_computing_method long_history \
--ignore_function_words

python3 calculate_ppl_acl.py \
--out_dir out_wikipedia_en \
--device cuda:0 \
--ppl_computing_method long_history \
--no-ignore_function_words

python3 calculate_ppl_acl.py \
--out_dir out_wikipedia_en \
--device cuda:0 \
--ppl_computing_method naive \
--ignore_function_words

python3 calculate_ppl_acl.py \
--out_dir out_wikipedia_en \
--device cuda:0 \
--ppl_computing_method naive \
--no-ignore_function_words