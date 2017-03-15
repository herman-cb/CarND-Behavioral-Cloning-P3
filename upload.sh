find . -name "left*.jpg" -delete
find . -name "right*.jpg" -delete
rm data.zip
zip -r data data
scp data.zip carnd@$1:/home/carnd/CarND-Behavioral-Cloning-P3

