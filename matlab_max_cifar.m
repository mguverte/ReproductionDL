clear all
close all
clc

% source this method is based on:
%https://arxiv.org/pdf/2102.07559.pdf
%% input x
cifar10_data=importdata('cifar10_data.png');
imge = cifar10_data(3:34,3:34,1:3); %rand(1,32*32*3);
imge=reshape(im2double(imge),[1,3072])';
%% optimize
% describe the optimization problem
net_imge = network_fnc_cif(imge);
nonlcon = @non_linear_cont;
options = optimoptions('fmincon','Display','iter','Algorithm','interior-point','MaxIterations',1000,'MaxFunctionEvaluations',1000000);
fun = @(delta)-(norm(network_fnc_cif(imge+delta)-net_imge));
x0 = ones(1,32*32*3)'.*10^-5;
% can be used to force decoded attacked signal into y \in [-1,1]
%A = [AB_ende; -AB_ende];
%b = [ones(344,1)-AB_ende*x_t_sinc;ones(344,1)+AB_ende*x_t_sinc];
%
delta = fmincon(fun,x0,[],[],[],[],[],[],nonlcon,options);

const_1 = norm(delta);

%% plot

est_or = network_fnc_cif(imge);
est_att = network_fnc_cif(imge+delta);
% plot(imge(2:2:344),imge(1:2:343))
% hold on
% plot(est_or(2:2:344),est_or(1:2:343))
% plot(est_att(2:2:344),est_att(1:2:343))
% legend("og_sinc", "decoded sinc", "attacked decoded sinc")


figure
image(img_size_convert(imge,32,32,3))
title('original image')
figure
image(img_size_convert(imge+delta,32,32,3))
title('attacked image')
figure
image(img_size_convert(network_fnc_cif(imge),32,32,3))
title('original decoded image')
figure
image(img_size_convert(network_fnc_cif(imge+delta),32,32,3))
title('attacked decoded image')

%% loglikelihood degradation
logdeg = mean(log(abs(est_or-est_att))./log(abs(est_or)));

