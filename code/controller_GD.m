% Real-time Neuro-Controller usig 2 layer MLP   
% Creator : Sohrab Rezaei
% Learning method: Gradian decent 
%% Initialization
close all; clear; clc;

%% Definiting Time 
sample_time=1e-4;                        % set sample time
final_time=100;                          % set simulation duration
time=0:sample_time:final_time;           % defining time
n_time=numel(time);                      % size of time data
time_p=linspace(0,final_time,n_time+3);

%% system Constants
x1_0=0; x2_0=0; x3_0=0;                               % initial conditions
J_t=0.03; B_m=1.1e-3; q_m=7.96e-7; C_f=0.104; P_s=1e7;  % system Constants
Beta_e=1.391e9; V_0=1.2e-4; C_im=1.69e-11; C_d=0.61;    % system Constants
W_s=0.008*pi; Ro=850; T_r=0.01; K_r=1.4e-4; K_q=1.66;   % system Constants
x1_sys=[x1_0,zeros(1,n_time+3)]';                       % state X1
x2_sys=[x2_0,zeros(1,n_time+3)]';                       % state X2
x3_sys=[x3_0,zeros(1,n_time+3)]';                       % state X3
y_sys=x1_sys;                                       % y=x1
e_c=zeros(n_time+1,1);                              % control erro
e_m=zeros(n_time+1,1);                              % identification error
min_u=0;                                        % minimum controller input
max_u=3;                                        % maximum controller input
min_y=-22;                                      % minimum system output
max_y=450;                                      % maximum system output

%% Y desired 
% yd=150*ones(n_time+3,1)+50*sin(2*pi*0.01*time_p');
yd_0=30*ones(30/sample_time,1); 
yd_1=100*ones(30/(sample_time),1);
tt=time_p(end-((40/(sample_time))+2):end);
yd_2=100*ones((40/(sample_time)+3),1)+50*sin(2*pi*0.05*tt)';
yd=[yd_0;yd_1;yd_2];
% yd=150*randn(n_time+3,1)+310;
yd=(yd-min_y)/(max_y-min_y);            % normalaized desired output
y_sys_n=(y_sys-min_y)/(max_y-min_y);    % normalaized system output

%% Controller
% network structure
n_c=5;  % number of controller's input
n1_c=5; % number of neurons in first layer
n2_c=1; % number of neurons in second layer
% hidden layer Controller
filename_w= 'Table_of_Parameters.xlsx';
w1_c=xlsread(filename_w,3,'B:F');   % W1 from train Gradian decent
eta1_c=0.07;                        % set eta1
% output layer
w2_c=xlsread(filename_w,3,'I:M');   % W1 from train Gradian decent 
eta2_c=0.075;                       % set eta2
output_c=zeros(n_time+3,1);

%% Identifier
% network structure
n_I=3;  % number of identifier's input
n1_I=4; % number of neurons in first layer
n2_I=1; % number of neurons in second layer
% hidden layer Controller
w1_I=xlsread(filename_w,2,'B:E');   % W1 from train Gradian decent
eta1_I=0.03;                        % set eta1
xb1=1;                              % Bias
% output layer
w2_I=xlsread(filename_w,2,'I:L');   % W2 from train Gradian decent
eta2_I=0.03;                        % set eta2
xb2=1;                              % Bias
output_I=zeros(n_time+3,1);

%% Real time system and controller
for j=3:n_time+1
    
        % Controller
        xi_1_c=yd(j);           
        xi_2_c= y_sys_n(j-1,1);
        xi_3_c=e_c(j-1);
        xi_4_c=e_c(j-2);
        input_c=[xb1,xi_1_c,xi_2_c,xi_3_c,xi_4_c];% set Inputs
        net1_c=w1_c*input_c';                     % set net1
        o1_c=logsig(net1_c);                      % o1 = 1/(1+exp(-net1))
        net2_c=w2_c*o1_c;                         % set net2
        o2_c=logsig(net2_c);                      % o2 = 1/(1+exp(-net2))
        u_c=(max_u-min_u)*o2_c+min_u;             % Controller output
        output_c(j)=u_c;
        
        % System
        x_dot1=(1/J_t)*(-B_m*x1_sys(j-1)+q_m*x2_sys(j-1)-q_m*C_f*P_s);
        x_dot2=(2*Beta_e/V_0)*(-q_m*x1_sys(j-1)-C_im*x2_sys(j-1)+C_d*W_s*x3_sys(j-1)*sqrt((1/Ro)*(P_s-x2_sys(j-1))));
        x_dot3=(1/T_r)*(-x3_sys(j-1)+(K_r/K_q)*u_c);
        delta_x1=x_dot1*sample_time;
        delta_x2=x_dot2*sample_time;
        delta_x3=x_dot3*sample_time;
        x1_sys(j)=x1_sys(j-1)+delta_x1;
        x2_sys(j)=x2_sys(j-1)+delta_x2;
        x3_sys(j)=x3_sys(j-1)+delta_x3;
        y_sys(j)=x1_sys(j);
        y_sys_n(j)=(y_sys(j)-min_y)/(max_y-min_y); % normalaize system output
        e_cc=yd(j)-y_sys_n(j);                     % system error
        e_c(j)=e_cc;
        
        % Identifier
        input_I=[xb1,output_c(j),output_c(j-1),y_sys_n(j-1,1)];  % set Inputs
        net1_I=w1_I*input_I';           % set net1
        o1_I=logsig(net1_I);            % o1 = 1/(1+exp(-net1))
        net2_I=w2_I*o1_I;               % set net2
        o2_I=logsig(net2_I);            % o2 = 1/(1+exp(-net2))
        y_I=o2_I;                       % identifier output
        output_I(j)=y_I;
        e_I=y_sys_n(j)-y_I;             % identifier error
        e_m(j)=e_I;
        
        % Gradian decent
        f1p_I=diag(o1_I.*(1-o1_I));         % d(o1)/d(net1) identifier
        f2p_I=diag(o2_I.*(1-o2_I));         % d(o2)/d(net2) identifier
        delta_w2_I=eta2_I*e_I*f2p_I*o1_I';  % delta w2 identifier
        delta_w1_I=eta1_I*e_I*f2p_I*(w2_I*f1p_I)'*input_I;% delata w1 identifier
        w2_I=w2_I+delta_w2_I;             % w1(k+1) identifier
        w1_I=w1_I+delta_w1_I;             % w2(k+1) identifier
        w_m_uc=w1_I(:,2)'; % coeffient of controller input (to calculate jacobian)
        f1pp_I=o1_I.*(1-o1_I);              % d(o1)/d(net1)
        J_m=w_m_uc*(f2p_I*(w2_I'.*f1pp_I)); % calculate Jacobian
        f1p_c=diag(o1_c.*(1-o1_c));         % d(o1)/d(net1) controller
        f2p_c=diag(o2_c.*(1-o2_c));         % d(o2)/d(net2) controller
        delta_w2_c=eta2_c*e_cc*J_m*f2p_c*o1_c';  % delta w2 controller
        delta_w1_c=eta1_c*e_cc*J_m*f2p_c*(w2_c*f1p_c)'*input_c;% delata w1 controller
        w2_c=w2_c+delta_w2_c;             % w1(k+1) controller
        w1_c=w1_c+delta_w1_c;             % w2(k+1) controller
        
end

%% plotting

figure(1);
% plotting results
subplot(2,2,1)
plot(y_sys_n(1:end-5),'-r');
hold on;
plot(yd(1:end-5),'-.b');
title('desired output and system output') 
legend('System','Desired')
grid
% zp = BaseZoom();
% zp.plot;
% zp = BaseZoom();
% zp.plot;
% zp = BaseZoom();
% zp.plot;
subplot(2,2,2)
plot(output_c(1:end-5,1),'-r');
title('controller output') 
grid

subplot(2,2,3)
plot(y_sys_n(1:end-5),'-r');
hold on
plot(output_I(1:end-5),'-.b');
title('Identifier') 
grid
legend('System','Indefier')
% zp = BaseZoom();
% zp.plot;
% zp = BaseZoom();
% zp.plot;
% zp = BaseZoom();
% zp.plot;
subplot(2,2,4)
plot(e_m(1:end),'-r');
hold on
plot(e_c(1:end),'-b');
grid
legend('Identifier error','Control error')
title('Error') 
% zp = BaseZoom();
% zp.plot;
% zp = BaseZoom();
% zp.plot;
% zp = BaseZoom();
% zp.plot;
sgtitle('Neuro-Controller Results')








