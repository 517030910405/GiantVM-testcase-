# cd model0
cp -r model model$1
cd model$1
python example.py --save-model --seed=1$1 --epochs=1