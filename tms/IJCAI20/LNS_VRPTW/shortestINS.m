%% ��Զ��������ʽ������С����Ŀ�������������Ԫ���ҳ���
%����removed                          ���Ƴ��Ĺ˿ͼ���
%����rfvc                             �Ƴ�removed�еĹ˿ͺ��final_vehicles_customer
%����L                                ��������ʱ�䴰
%����a                                �˿�ʱ�䴰
%����b                                �˿�ʱ�䴰
%����s                                ����ÿ���˿͵�ʱ��
%����dist                             �������
%����demands                          ������
%����cap                              ���������
%���fv                               ��removed������Ԫ�� ��Ѳ���������������Ԫ��
%���fviv                             ��Ԫ��������ĳ���
%���fvip                             ��Ԫ��������ĳ���������
%���fvC                              ��Ԫ�ز������λ�ú�ľ�������
function [sv,sviv,svip,svC]=shortestINS(removed,rfvc,L,a,b,s,dist,demands,cap )
nr=length(removed);                   %���Ƴ��Ĺ˿͵�����
outcome=zeros(nr,3);
for i=1:nr
    %[������� �������� ��������]
    [civ,cip,C]= cheapestIP( removed(i),rfvc,L,a,b,s,dist,demands,cap);
    outcome(i,1)=civ;
    outcome(i,2)=cip;
    outcome(i,3)=C;
end
[mc,mc_index]=min(outcome(:,3));
temp=outcome(mc_index,:);
sviv=temp(1,1);
svip=temp(1,2);
svC=temp(1,3);
sv=removed(mc_index);

end
