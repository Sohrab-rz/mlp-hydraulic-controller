% Neuro-Controller usig 2 layer MLP   
% Creator : Sohrab Rezaei
% Learning method: Gradian decent 
%% Initialization
close all;
clear;
clc;
epoch=100;
%% Time 
sample_time=1e-3;
time=0:sample_time:100; 
n_time=numel(time);

%% system Constants
x1_0=2; x2_0=1; x3_0=4;                               % initial conditions
J_t=0.03; B_m=1.1e-3; q_m=7.96e-7; C_f=0.104; P_s=1e7;  % system Constants
Beta_e=1.391e9; V_0=1.2e-4; C_im=1.69e-11; C_d=0.61;    % system Constants
W_s=0.008*pi; Ro=850; T_r=0.01; K_r=1.4e-4; K_q=1.66;   % system Constants
x1_sys=[x1_0,zeros(1,n_time+3)]';
x2_sys=[x2_0,zeros(1,n_time+3)]';
x3_sys=[x3_0,zeros(1,n_time+3)]';
y_sys=x1_sys;

%% input signal
freq1=logspace(-2,1,3);
freq2=logspace(1,2,5);
freq3=logspace(2,3,8);
freq=[freq1,freq2,freq3];
AA=[0.2*ones(1,numel(freq1)),0.015*ones(1,numel(freq2)),0.008*ones(1,numel(freq3))];
ft=2*pi*freq'*time;
uu=sin(ft)+cos(ft-2*pi/3);
u_in=AA*uu+1.5;
% u_in=randn(1,numel(u_in));


for j=3:n_time
        % System
        x_dot1=(1/J_t)*(-B_m*x1_sys(j-1)+q_m*x2_sys(j-1)-q_m*C_f*P_s);
        x_dot2=(2*Beta_e/V_0)*(-q_m*x1_sys(j-1)-C_im*x2_sys(j-1)+C_d*W_s*x3_sys(j-1)*sqrt((1/Ro)*(P_s-x2_sys(j-1))));
        x_dot3=(1/T_r)*(-x3_sys(j-1)+(K_r/K_q)*u_in(j));
        delta_x1=x_dot1*sample_time;
        delta_x2=x_dot2*sample_time;
        delta_x3=x_dot3*sample_time;
        x1_sys(j)=x1_sys(j-1)+delta_x1;
        x2_sys(j)=x2_sys(j-1)+delta_x2;
        x3_sys(j)=x3_sys(j-1)+delta_x3;
        y_sys(j)=x1_sys(j);
end
%% normalize
y_sys=y_sys(3:n_time);
u_in=u_in(3:n_time);
min_u=0;
max_u=3;
min_y=-22;
max_y=450;
y_sys=(y_sys-min_y)/(max_y-min_y);
u_in=(u_in-min_u)/(max_u-min_u);

%% Time 
time=time(3:n_time); 
n_time=numel(time);
figure(1);
plot(time,u_in)

%% data
data=[u_in(1:end-1)',u_in(2:end)', y_sys(1:end-1),y_sys(2:end)];
sizeData=size(data);
ro=sizeData(1);     % number of data sets
co=sizeData(2);     % number of inputs+target
train_rate=0.7;     % set train rate
estimate_rate=0.30; % set validation rate
test_rate=0.00;     % set test rate
num_of_train=round(train_rate*ro);
num_of_validation=round(estimate_rate*ro);
num_of_test=ro-(num_of_train+num_of_validation);
data_train=data(1:num_of_train,:);   % set train data
data_validation=data((num_of_train+1):(num_of_train+num_of_validation),:);% set validation data
data_test=data(ro-num_of_test:ro,:); % set test data 


%% Identifier
%% network structure
n=co-1; % number of inputs except X bias
n1=4; % number of cells in first layer
n2=1; % number of cells in second layer

%% hidden layer
L_b_w1=-1; % w1 lower bound
U_b_w1=1;  % w1 upper bound
w1=L_b_w1+(U_b_w1-L_b_w1)*rand(n1,n+1); % uniformly distributed random between variable [L_b_w1,U_b_w1] 
eta1=0.05; % set eta1
xb1=1;     % Bias

%% output layer
L_b_w2=-1; % w2 lower bound
U_b_w2=1;  % w2 upper bound
w2=L_b_w2+(U_b_w2-L_b_w2)*rand(n2,n1); % uniformly distributed random between variable [L_b_w2,U_b_w2] 
eta2=0.06; % set eta2
 
%% training the NN
for j = 1:epoch
 % train
    for i = 1:num_of_train
        x=[xb1 data_train(i,1:n)];  % set Inputs
        target=data_train(i,co);    % Set Target
        net1=w1*x';                 % set net1
        o1=logsig(net1);            % o1 = 1/(1+exp(-net1))
        net2=w2*o1;                 % set net2
        o2=logsig(net2);                    % o2 = net2
        e=target-o2;                % e(k)
        output_train(i,1)=o2;       % output(k)
        error_train(i,1)=e;
    % Gradian decent
        f1p=diag(o1.*(1-o1));       % d(o1)/d(net1)
        f2p=diag(o2.*(1-o2));
        delta_w1=eta1*e*f2p*(w2*f1p)'*x;% delata w1
        w1=w1+delta_w1;             % w1(k+1)
        delta_w2=(eta2*e*f2p*o1)';      % delata w2
        w2=w2+delta_w2;             % w2(k+1)
        
       
    end
    e_mse_train(j,1)=mse(error_train);
    
    % test
    for i = 1:num_of_validation
        x=[xb1 data_validation(i,1:n)]; % set Inputs
        target=data_validation(i,co);   % Set Target
        net1=w1*x';                 
        o1=logsig(net1);
        net2=w2*o1;
        o2=logsig(net2);
        e=target-o2;
        output_validation(i,1)=o2;
        error_validation(i,1)=e;
    end
    e_mse_validation(j,1)=mse(error_validation);
    
    % this function prevents validation error to increase
    if j > 50
        c2=e_mse_validation(j,1)-e_mse_validation(j-1,1); 
        if c2 > 0
            disp('validation error is increasing')
            break;
        end
    end
    
  % plot
    
  figure(2);
  subplot(2,2,1),plot(data_train(:,co),'-r');
  title('Data train')
  hold on;
  subplot(2,2,1),plot(output_train,'-b');
  
  hold off;
  subplot(2,2,2),semilogy(e_mse_train(1:j,1),'-r');
  title('Error train')
  hold off;
  
  subplot(2,2,3),plot(data_validation(:,co),'-r');
  hold on;
  subplot(2,2,3),plot(output_validation,'-b');
  title('Data validation')
  hold off;
  subplot(2,2,4),plot(e_mse_validation(1:j,1),'-r');
  title('Error validation')
  sgtitle('Gradian decent - Identification results')
  hold off;
  
%   pause(0.05);
    
end



