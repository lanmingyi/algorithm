E=dataset(1,5);                                                %配送中心时间窗开始时间
L=dataset(1,6);                                                %配送中心时间窗结束时间
% cap=200;                                                    %车辆负荷
vertexs=dataset(:,2:3);                                        %所有点的坐标x和y
customer=vertexs(2:end,:);                                  %顾客坐标
cusnum=size(customer,1);                                    %顾客数
vecnum=cusnum;                                              %车辆数
demands=dataset(2:end,4);                                      %需求量
a=dataset(2:end,5);                                            %顾客时间窗开始时间[a[i],b[i]]
b=dataset(2:end,6);                                            %顾客时间窗结束时间[a[i],b[i]]
s=dataset(2:end,7);                                            %客户点的服务时间