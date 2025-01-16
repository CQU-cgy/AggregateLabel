fashion_mnist_test = load("fashion_mnist_train.csv");

test_img = fashion_mnist_test(:,2:785);
test_label = fashion_mnist_test(:,1);

img = test_img(2,:);
img = reshape(img,28,28);
gray = mat2gray(img);
gray = kron(gray',ones(16,16));

imshow(gray)