python3 calculate_ppl_acl.py \
--model_dir out_wikipedia_en \
--device cuda:0 \
--computing_method long_history \

python3 calculate_ppl_acl.py \
--model_dir out_wikipedia_en \
--device cuda:0 \
--mimimum_context_length 1023 \
--computing_method long_history \

python3 calculate_ppl_acl.py \
--model_dir out_wikipedia_en \
--device cuda:0 \
--computing_method long_history \

python3 calculate_ppl_acl.py \
--model_dir out_wikipedia_en \
--device cuda:0 \
--computing_method naive \

python3 calculate_ppl_acl.py \
--model_dir out_openwebtext \
--device cuda:0 \
--computing_method long_history \
