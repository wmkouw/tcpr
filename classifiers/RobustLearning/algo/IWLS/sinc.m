function y=sinc(x)

y=ones(size(x));
i=find(x);
y(i)=sin(pi*x(i))./(pi*x(i));
