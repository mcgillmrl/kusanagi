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

model.fcn = @gp2;                % function for GP predictions
model.train = @train;             % function to train dynamics model
trainOpt = [300 500];                % defines the max. number of line searches

model.inputs  = [X(:,nangi), Xa];
model.targets = Y;
%model = model.train(model, [], trainOpt);  %  train dynamics GP
model.hyp = [1.488644640716878,   1.376880336167960,   1.405309748390113,  0.794514565057032,   1.264445698631373,   3.587939447644951,   0.378659570401003,  -2.107781681104368,  -2.670827107680510;
             1.487738978825120,   1.377055841019811,   1.405875137232056,  0.794717434510711,   1.267984365286328,   3.540119863989067,   0.381918802434586,  -1.415237559412543,  -1.979235545504115;
             1.487908336813194,   1.379300719560770,   1.404292670414302,  0.793524475430800,   1.267560010517307,   3.595231469390403,   0.377971363284628,  -1.011282163586751,  -1.573726951186254]';
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
