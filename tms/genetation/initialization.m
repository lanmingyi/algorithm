function path = initialization(transport_time,customer,vehicle,espo)
path(espo).infor = []; 
espoo = 1;
while espoo <= espo
    customers = customer;
    vehicles = vehicle;
    
    %������·����ʼ��
    select(1).path=0;
    select(2).path=0;
    select(3).path=0;
    select(4).path=0;
    select(5).path=0;
    select(6).path=0;
    select(7).path=0;
    select(1).num=1;
    select(2).num=2;
    select(3).num=3;
    select(4).num=4;
    select(5).num=5;
    select(6).num=6;
    select(7).num=7;
    select(1).price=0.2;
    select(2).price=0.2;
    select(3).price=0.3;
    select(4).price=0.15;
    select(5).price=0.15;
    select(6).price=0.1;
    select(7).price=0.1;
    
    %
    cus = false ;
    while cus ==  false
        % ȡ�����������Ŀͻ���������������ֵΪ�ա�
        max_demand = max(customers(2,:)) ;
        [i,j] = find(customers == max_demand);
        customer_i = customers(1,j(1));
        customers(: ,j(1)) = [];
        vel = false;
        %���гɱ��ļ���
        while vel == false
            % ���ѡȡ����
            a = floor(rand(1,1)*size(vehicles,2)+1);
            num = select(a).path(1,end);
            %ȡ����������˵ص�ʱ��
            t = transport_time(num+1,customer_i + 1);
            % �жϳ����Ƿ�����������������ʱ���Ҫ��
            if vehicles(2,a) >= max_demand && vehicles(3,a) > t
                % ���Ϊ�棬����·��������ʱ�䣬����ʣ��������,�� vel = true
                select(a).path(end + 1) = customer_i;
                vehicles(2,a) = vehicles(2,a) - max_demand;
                vehicles(3,a) = vehicles(3,a) - t;
                vel = true;
            end
        end
        if isempty(customers)
            cus = true;
        end
    end
 % ��·�������ᴿ���õ��ĳ���ȥ��
    for pat = 1:size(select,2)
        if select(pat).path(end) == 0
            continue
        else
            for path_num = 1:size(select(pat).path,2)
                path(espoo).infor(1,end+1) = select(pat).num;
                path(espoo).infor(2,end) = select(pat).price;
                path(espoo).infor(3,end) = select(pat).path(1,path_num);
            end
        end
    end
    espoo =espoo + 1;
end
% �������·���������

