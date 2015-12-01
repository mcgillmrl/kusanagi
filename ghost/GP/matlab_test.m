clear all; 
close all;

rand('twister', 31337)
try
  rd = '/home/juancamilog/workspace/antoinette/pilcoV0.9/';
  addpath([rd 'base'],[rd 'util'],[rd 'gp'],[rd 'control'],[rd 'loss']);
catch
end

n = 100
n_test = 10
D = 2
E = 3

f = @(x) exp(-500*sum(0.001*x.^2,2));
%m0 = randn(1,D)'
%S0 = randn(D,D)'
%S0 = eye(D);

X = 10*(rand(D,n)' - 0.5)
pause;
Y = zeros(E,n)';
for i=1:E
    Y(:,i) = i*f(X) + 0.01*(rand(1,n)' - 0.5);
end
Y
pause;

model.fcn = @gp0d;                % function for GP predictions
model.train = @train;             % function to train dynamics model
trainOpt = [300 500];                % defines the max. number of line searches

model.inputs  = X;
model.targets = Y;
model = model.train(model, [], trainOpt);  %  train dynamics GP
exp(model.hyp')


Xtest = 10*(rand(D,n_test)' -0.5)
pause;
Ytest = zeros(E,n_test)';
for i=1:E
    Ytest(:,i) = i*f(Xtest) + 0.01*(rand(1,n_test)' - 0.5)
end
Ytest
pause;

for i=1:size(Ytest,1)
    disp('---')
    disp(['x: ', num2str(Xtest(i,:)),', y: ',num2str(Ytest(i,:))])
    [M, S, V] = model.fcn(model, Xtest(i,:)', zeros(D));
    M
    S
end
