
guidance_configs = (
    "no_guide"
    "z_pos_10"
    #"z_pos_100"
    # "z_pos_500"
    # "z_neg_10"
    # "z_neg_100"
    # "z_neg_500"
)

for config in ${guidance_configs[@]} ; do
    python eval.py -c epoch=1750-test_mean_score=1.000.ckpt -o ${config}_data -n 5 -g guidance_configs/${config}.json 
done




