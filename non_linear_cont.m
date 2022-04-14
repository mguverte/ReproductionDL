function [c, ceq] = non_linear_cont(delta)
c_d = 20;
c(1) = norm(delta)-c_d;
%c(2) = max((x_sinc+delta).^2)-1;
ceq = [];
end

