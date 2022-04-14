function y = img_size_convert(x, a, b, c)
len_x = length(x);
y = zeros(a,b,c);
for i = 0:c-1
    x_tmp = x(i*len_x/c+1:(i+1)*len_x/c);
    len_x_tmp = length(x_tmp);
    for j = 0:b-1
        y(j+1,:,i+1) = x_tmp(j*len_x_tmp/a+1:(j+1)*len_x_tmp/a);
    end
end
end

