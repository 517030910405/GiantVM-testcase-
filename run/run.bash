rm mnist_cnn*
rm mytime.txt
python readtime.py > mytime.txt
bash runs.bash 0&
bash runs.bash 1&
bash runs.bash 2& 
bash runs.bash 3

python ensemble.py --nameFrom=0 --nameTo=4

python readtime2.py
python readtime2.py>> resultTime.txt