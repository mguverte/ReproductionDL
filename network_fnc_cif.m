function [x_est] = network_fnc_cif(x)
% convert and load all the weights and biases generated by the learner
addpath('Weights2')
load('WeightsR2-sigma-1_0.mat')



encode1 = cell2mat(weights(1));
encode2 = cell2mat(weights(2));
encode3 = cell2mat(weights(3));
encodeu = cell2mat(weights(4));
load('WeightsL2-sigma-1_0.mat')
decode1 = cell2mat(weights(1));
decode2 = cell2mat(weights(2));
decode3 = cell2mat(weights(3));
decode4 = cell2mat(weights(4));



load('biasR2-sigma-1_0.mat')
bias_en1 = cell2mat(bias(1));
bias_en2 = cell2mat(bias(2));
bias_en3 = cell2mat(bias(3));
bias_enu = cell2mat(bias(4));
%bias_enu = bias_enu(1:256);
load('biasL-sigma-1_0.mat')
bias_de1 = transpose(cell2mat(bias(1)));
bias_de2 = transpose(cell2mat(bias(2)));
bias_de3 = transpose(cell2mat(bias(3)));
%bias_de4 = zeros(3072,1);
%bias_de4(1:64) = transpose(cell2mat(bias(4)));
bias_de4 = transpose(cell2mat(bias(4)));
% bias_en1 = zeros(1,256);
% bias_en2 = zeros(1,256);
% bias_en3 = zeros(1,256);
% bias_enu = zeros(1,64);
% bias_de1 = zeros(256,1);
% bias_de2 = zeros(256,1);
% bias_de3 = zeros(256,1);
% bias_de4 = zeros(3072,1);

z = transpose(encodeu*sig_imp(encode3*sig_imp(encode2*sig_imp(encode1*x+bias_en1)+bias_en2)+bias_en3)+bias_enu);
x_est = transpose(decode4*sig_imp(decode3*sig_imp(decode2*sig_imp(decode1*z+bias_de1)+bias_de2)+bias_de3)+bias_de4);

end

