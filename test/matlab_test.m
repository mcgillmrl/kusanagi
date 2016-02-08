clear all; 
close all;
format long;
rand('twister', 31337)
py.numpy.random.seed(int64(31337));
n = 1000;
n_test = 20;
D = 5;
E = 3;
angi = [1,2]

f = @(x) exp(-500*sum(0.0001*x.^2,2));
nangi = setdiff(1:D,angi);
X = 10*(pyrand(D,n)' - 0.5);
Y = zeros(n,E);
Xa = zeros(n,2*length(angi));
for i=1:E
    Y(:,i) = i*f(X) + 0.01*(pyrand(1,n) - 0.5)';
end
kk=0.01*conv2([1,2,3,2,1],[1;2;3;2;1])/9.0;

for i=1:size(Y,1)
    ss = conv2(eye(D),kk,'same');
    [Xa(i,:),~,~] = gTrig(X(i,:),ss,angi);
end

model.fcn = @gp0;                % function for GP predictions
model.train = @train;             % function to train dynamics model
trainOpt = [300 500];                % defines the max. number of line searches

model.inputs  = [X(:,nangi), Xa];
model.targets = Y;
model = model.train(model, [], trainOpt);  %  train dynamics GP

Xtest = 10*(pyrand(D,n_test)' -0.5);
Ytest = zeros(E,n_test)';
Xatest = zeros(n_test,2*length(angi));
for i=1:E
    Ytest(:,i) = i*f(Xtest) + 0.01*(pyrand(1,n_test) - 0.5)';
end

for i=1:size(Ytest,1)
    ss = conv2(eye(D),kk,'same');
    [Xatest(i,:),~,~] = gTrig(Xtest(i,:),ss,angi);
    XX_ = [Xtest(i,nangi), Xatest(i,:)];
    disp(['x: ', num2str(XX_),', y: ',num2str(Ytest(i,:))])
    ss = conv2(eye(size(XX_,2)),kk,'same');
    [M, S, V] = model.fcn(model, XX_', ss);
    M'
    S
    V
    disp('---')
end
