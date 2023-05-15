clear;
clc;
transport_time = [0,2,3,2,3,3,3,4,4,5,3;...
    2,0,3,3,2,2,3,3,5,5,3;...
    3,3,0,4,3,2,1,2,4,3,2;...
    2,3,4,0,4,4,5,5,6,6,4;...
    3,2,3,4,0,2,2,3,5,5,2;...
    3,2,2,4,2,0,2,2,5,5,1;...
    3,3,1,5,2,2,0,1,4,4,1;...
    4,3,2,5,3,2,1,0,5,3,1;...
    4,5,4,6,5,5,4,5,0,5,5;...
    5,5,3,6,5,5,4,3,5,0,4;...
    3,3,2,4,2,1,1,1,5,4,0]; % ������ҵ������ʱ���


customers_demand =[1,2,3,4,5,6,7,8,9,10;3000,10000,5000,2000,2000,10000,8000,5000,7000,20000]; % �ͻ��Ի����������
customers_time = [1,2;2,4;3,5;4,8;3,10;11,15;3,6;7,12;4,10;2,7]'; % �ͻ���������̵�ʱ�䴰Լ��
customer = [customers_demand;customers_time];   %�ͻ��Ļ�����Ϣ���ͻ���ţ���������ʱ�䴰Լ����

num_of_vehicles = [1,2,3,4,5,6,7]; %�����Ǹ����
situation_of_vehicles = [15000,15000,2000,25000,25000,40000,40000];%������ҵ�������������
time_of_vehicles = ones(1,7)*30; %�������������ʱ��
vehicle = [num_of_vehicles;situation_of_vehicles;time_of_vehicles];%��һ��Ϊ������ţ��ڶ���Ϊ���أ�������Ϊʣ��ʱ��

number_of_car = size(num_of_vehicles,2); %ӵ����������
cost_sortage = 0.44; % ÿ�ֵ����͵���ÿ�촢��ɱ�
deadline_cost = 4000; % �ӳ��͵�������ɱ�����λÿ��


espo = 50  ;  %��Ⱥ����
popsize = espo;
Generationnmax=3;  %������

%% ������ʼ��Ⱥ,���ص�selectΪ����·����������š���λ�˷ѵĽṹ��
path = initialization(transport_time,customer,vehicle,espo);


%% ������Ӧ��,������Ӧ��Fitvalue���ۻ�����cumsump
new_path =fitnessfun(path,transport_time,customer,cost_sortage,deadline_cost,number_of_car) ;
%% 
Generation=1;
while Generation < Generationnmax+1
   for j=1:2:popsize
      %ѡ�����
      [father_1,father_2] = selection(new_path);
      %�������
      [path_1_infor_1,path_1_infor_2]=crossover(father_1,father_2,transport_time,customer,number_of_car);
      %�������
      snnew_1=mutation(path_1_infor_1,transport_time,number_of_car,vehicle,customer);
      new_f(j).infor = snnew_1;
      snnew_2=mutation(path_1_infor_2,transport_time,number_of_car,vehicle,customer);
      new_f(j+1).infor = snnew_2;
      j
   end
   path = new_f;  %�������µ���Ⱥ 
   %��������Ⱥ����Ӧ��   
   new_path = fitnessfun(path,transport_time,customer,cost_sortage,deadline_cost,number_of_car)
   %��¼��ǰ����õ���Ӧ�Ⱥ�ƽ����Ӧ��
   Generation = Generation+1
end
new_path