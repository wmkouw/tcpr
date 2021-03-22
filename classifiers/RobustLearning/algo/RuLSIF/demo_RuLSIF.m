% demo_RuLSIF.m
%
% (c) Makoto Yamada & Masashi Sugiyama, Department of Compter Science, Tokyo Institute of Technology, Japan.
%     yamada@sg.cs.titech.ac.jp, sugi@cs.titech.ac.jp,     http://sugiyama-www.cs.titech.ac.jp/~sugi/software/RuLSIF/

clear all
%profile clear
%profile on;
seed = 1;
rand('state',seed);
randn('state',seed);

%%%%%%%%%%%%%%%%%%%%%%%%% Generating data
d=1;

dataset=1;
switch dataset
%Same distribution
case 1
  n_de=1000;
  n_nu=1000;
  mu_de=1;
  mu_nu=1;
  sigma_de=1/2;
  sigma_nu=1/2;
  legend_position=1;
%Different distribution
 case 2
  n_de=200;
  n_nu=1000;
  mu_de=1;
  mu_nu=1.5;
  sigma_de=1/4;
  sigma_nu=1/4;
  legend_position=2;
end

x_de=mu_de+sigma_de*randn(d,n_de);
x_nu=mu_nu+sigma_nu*randn(d,n_nu);

alpha = 0.5;

x_disp=linspace(-0.5,3,100);
p_de_x_disp=pdf_Gaussian(x_disp,mu_de,sigma_de);
p_nu_x_disp=pdf_Gaussian(x_disp,mu_nu,sigma_nu);
w_x_disp=p_nu_x_disp./(alpha*p_nu_x_disp + (1 - alpha)*p_de_x_disp);

p_de_x_de=pdf_Gaussian(x_de,mu_de,sigma_de);
p_nu_x_de=pdf_Gaussian(x_de,mu_nu,sigma_nu);
w_x_de=p_nu_x_de./(alpha*p_nu_x_de + (1 - alpha)*p_de_x_de);

%%%%%%%%%%%%%%%%%%%%%%%%% Estimating density ratio
%[wh_x_de,wh_x_disp]=uLSIF(x_de,x_nu,x_disp);

tic
[PE,wh_x_de,wh_x_disp]=RuLSIF(x_nu,x_de,x_disp,alpha,[],[],[],5);
toc

figure(1) 
clf   
hold on  
set(gca,'FontName','Helvetica') 
set(gca,'FontSize',20)
plot(x_disp,p_nu_x_disp,'k-','LineWidth',2);
plot(x_disp,p_de_x_disp,'b--','LineWidth',5);
plot(x_disp,w_x_disp,'r-','LineWidth',2);

h12 = xlabel('$x$');
axis([min(x_disp) max(x_disp) 0 4]);
set(h12,'interpreter','latex');
set(gcf,'PaperUnits','centimeters');
set(gcf,'PaperPosition',[0 0 12 9]);

sstr1 = sprintf('$p(x)$');
sstr2 = sprintf('$q(x)$');
sstr3 = sprintf('$r_{%1.1f}(x)$', alpha);
h21 = legend(sstr1,sstr2,sstr3,legend_position);
set(h21,'interpreter','latex');
title(sprintf('(Estimated rPE) = %g',PE))

print('-dpng',sprintf('density%g',dataset))

figure(2)
clf
hold on
set(gca,'FontName','Helvetica')
set(gca,'FontSize',20)
plot(x_disp,w_x_disp,'r-','LineWidth',2)
plot(x_disp,wh_x_disp,'g--','LineWidth',2)
plot(x_de,wh_x_de,'bo','LineWidth',1,'MarkerSize',8)
format short;
sstr1 = sprintf('$r_{%1.1f}(x)$', alpha);
sstr2 = sprintf('$\\widehat{r}_{%1.1f}(x)$', alpha);
sstr3 = sprintf('$\\widehat{r}_{%1.1f}(x^{\\mathrm{de}}_j)$', alpha);
h21 = legend(sstr1,sstr2,sstr3,legend_position);
set(h21,'interpreter','latex');
h22 = xlabel('$x$');
set(h22,'interpreter','latex');
set(gcf,'PaperUnits','centimeters');
set(gcf,'PaperPosition',[0 0 12 9]);

title('Estimated relative density-ratio');

print('-depsc',sprintf('importance%g',dataset))


