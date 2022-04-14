function y = sig_imp (x)
y = zeros(length(x),1);
for i = 1:length(x)
    y(i,1) = 1/(1+exp(-x(i)));
end

end