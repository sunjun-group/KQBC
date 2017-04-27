% new query data generated.
%function [new_x,X_train,w1,w2] = newx(X_train, Y_train, coef, selection, dim)

tol = 1e-10;   %tolerance for the factor function error,
K = X_train*X_train';
samp_num = length(Y_train);
D = int16(dim(1));

[u,s] = schur(K(selection,selection));
s = diag(s);
I = (s>tol);
A = u(:,I)*diag(s(I).^-0.5);
restri = diag(Y_train(selection))*K(selection,selection)*A;

co1 = pinv(A)*coef(selection);
co2 = hit_n_run(co1,restri,100);
co1 = hit_n_run(co2,restri,100);
aco1=A*co1;
aco2=A*co2;
w1 = X_train(selection,:)'*(A*co1);
w2 = X_train(selection,:)'*(A*co2);

i=0;
n=0;
while(i==0&&n<50000)
%     new_x = randi(10000, [2, 1]);
    new_x = randi(10000, [D, 1]);
    
    %  	    x1=randi([0,100000]);
    %  	    x2=randi([0,100000]);
    %         new_x=[x1,x2];
    
    % 	    if((x1*w1(1)+x2*w1(2)) * (x1*w2(1)+x2*w2(2)) < 0)
    %         if ((new_x .* w1) * (new_x .* w2) < 0)
    pred1 = 0;
    pred2 = 0;
    j = 1;
    while (j <= D)
        pred1 = pred1 + new_x(j) * w1(j);
        pred2 = pred2 + new_x(j) * w2(j);
        j = j + 1;
    end
    if (pred1 * pred2 < 0)
        for rowi=1:samp_num;
            tmpt = X_train(rowi);
%             if (tmpt == new_x)
                %                 if(tmpt(1)==x1)
                %                     if(tmpt(2)==x2)
%                 break;
                %                     end
%             end
            if(rowi == samp_num)
                i=i+1;
                X_train = [X_train; new_x];
            end
        end
    end
    n=n+1;
end
if(n==50000)
    disp('No reasonable pairs!');
end
