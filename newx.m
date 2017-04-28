% new query data generated.
function [new_x,X_train,w1,w2] = newx(X_train, Y_train, coef, selection)  

	tol = 1e-10;   %tolerance for the factor function error,
	K = X_train*X_train'%; % [n*D] * [D*n] --> [n * n]
	%X_row = length(X_train(:, 1))
	%X_column = length(X_train(1, :)) % dim
	samp_num = length(Y_train);

	[u,s] = schur(K(selection,selection)) %u->[n*n] s->[n*n]
	%u_row = length(u(:, 1))
	%u_col = length(u(1, :))
	%s_row = length(s(:, 1))
	%s_col = length(s(1, :))

	s = diag(s) % n*1 % pick up the elements in the diagnal
	I = (s>tol) % n*1
	%ui = u(:, I) 
	si = s(I)
	si1_2 = s(I).^-0.5 % #*1
	Sidiag = diag(si1_2) % # * #
    Usub = u(:, I)
	A = u(:,I)*diag(s(I).^-0.5) % n*D     [n*#]  * [#*#] --> [n*#] 
	%selection
	%Y_train(selection)
	Ydiag = diag(Y_train(selection))
    kselection = K(selection, selection)
	restri = diag(Y_train(selection))*K(selection,selection)*A % [n*n] * [n*n]  * [n*#] --> n*#
	%restri_row = length(restri(:, 1))
	%restri_col = length(restri(:, 1))

	pinvA = pinv(A) % #*n
	coef_selection = coef(selection) % n * 1
	co1 = pinv(A)*coef(selection) % # * 1
	co2 = hit_n_run(co1,restri,6)
	co1 = hit_n_run(co2,restri,6)
	aco1=A*co1;  % n * 1
	aco2=A*co2;  % n * 1
	w1 = X_train(selection,:)'*(A*co1) % [D*n] * [n*1]  --> [D*1]
	w2 = X_train(selection,:)'*(A*co2)
	%w1_row = length(w1(:, 1))
	%w1_col = length(w1(1, :))

	i=0;
	n=0;
	while(i==0&&n<50000)
	    x1=randi([0,100000]);
	    x2=randi([0,100000]);

	    if((x1*w1(1)+x2*w1(2)) * (x1*w2(1)+x2*w2(2)) < 0)
		for rowi=1:samp_num;
		    tmpt = X_train(rowi);
		    if(tmpt(1)==x1)
		        if(tmpt(2)==x2)
				break;
		        end
		    end
		    if(rowi == samp_num)
		        i=i+1;
			new_x=[x1,x2];
		        X_train = [X_train; new_x];
		    end
		end
	    end
	    n=n+1;
	end
	if(n==50000)
	    disp('No reasonable pairs!');
	end
end