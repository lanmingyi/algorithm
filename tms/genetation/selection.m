%�ӳ�������Ⱥѡ�����, �������ƴ洢Ϊselection.m
function [father_1,father_2]=selection(new_path)
%����Ⱥ��ѡ����������
a = false;
father_num = 0; %��ʼ����������
father_1 = [];
father_2 = [];% ��ʼ������
while a == false
    r = floor(rand * 49 + 1); 
    if new_path(r).series == 0;
        if father_num == 0
            father_1 = new_path(r);
            father_num = father_num + 1;
        elseif father_num == 1
            father_2 = new_path(r);
            a = true;
         end
    end
end
