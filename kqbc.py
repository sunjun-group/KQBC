#!/usr/local/bin/python

import os, sys
from numpy import *
from scipy import linalg

"""
Inputs:
  X_train - Training instances
  Y_train - Training labels (values +1 and -1).
  T       - Number of random walk steps to make when selecting a
            random hypothesis.
  kernel  - Type of kernel to be used. Possible values are 'Linear', 'Poly'.
"""


def hit_n_run(x, A, T):
	dim = len(x)
	x = x[:]
	u = random.standard_normal((T, dim))
	#u = array([[-0.5901215, -1.1857805], [1.3043520, 0.5238996], [-0.9137054, -1.9349324], [-1.1734301, 1.9676968], [0.1669702, -0.5605912], [0.6431607, 1.8868448], [-1.5397598, 1.0546558], [0.0444157, -0.2146521], [0.3021594, 0.8111635], [-0.9411494, -0.0074194]])
	Au = dot(u, A.transpose())
	#print 'A = ', A
	#print Au
	nu = sum(power(u, 2), axis=1)
	l = random.rand(T, 1)
	#l = array([[0.25794], [0.93939], [0.54765], [0.96549], [0.34526], [0.12297], [0.75985], [0.27344], [0.39572], [0.77695]])
	
	for t in range(0, T):
		Ax = dot(A, x)
		#print 'Ax = ', Ax
		#print 'Au[t, :] = ', Au[t, :]
		ratio = -Ax.transpose() / Au[t, :]
                #print 'ratio = ', ratio
		I = (Au[t, :] > 0)
		#print 'I', I
		#print 'ratio[I] = ', ratio[I]
		mn = ratio[I].max() if shape(ratio[I])[1] > 0 else -sys.float_info.max
		#print 'mn = ', mn

		I = (Au[t, :] < 0)
		mx = ratio[I].min() if shape(ratio[I])[1] > 0 else sys.float_info.max
		#print 'mx = ', mx

                #print dot(nu[t], power(linalg.norm(x), 2) - 1)
		disc = power(dot(x.transpose(), u[t, :].transpose()), 2) - dot(nu[t], power(linalg.norm(x), 2) - 1)
		#print disc
                disc = disc[0]
                
		if disc < 0:
                        print 'negative disc {}. Probably x is not a feasible point.'.format(disc)
                        disc = 0

                hl = (-dot(x.transpose(), u[t, :].transpose()) + sqrt(disc)) / nu[t]
                ll = (-dot(x.transpose(), u[t, :].transpose()) - sqrt(disc)) / nu[t]

                xx = min(hl, mx)
                nn = max(ll, mn)

                #print shape(u[t, :].reshape(dim, 1))
                #print shape((nn + dot(l[t], xx - nn)).reshape(1, 1))
                #print dot(matrix(u[t, :]), matrix((nn + dot(l[t], xx - nn))))
                x = x + dot(u[t, :].reshape(dim, 1), (nn + dot(l[t], xx - nn)).reshape(1, 1))

        return x

tol = 1e-10   #tolerance for the factor function
kernel = 'Linear'

#X_train = [[0.9075111292951604, 0.24594145529546663, 1], [0.15278353125690602, 0.5724453847954805, 1], [0.9113121783002992, 0.5255336942828444, 1], [0.20723891545941453, 0.11195877723091241, 1], [0.7291033618567253, 0.1342145119788115, 1], [0.39020126002110367, 0.8926738122553486, 1], [0.8778819468166289, 0.1133677893144589, 1], [0.5083192308366595, 0.8604050392224345, 1], [0.9625811860712793, 0.8358437784413942, 1], [0.354859156315527, 0.9020153489314061, 1], [0.8688545369350471, 0.48203972534405737, 1], [0.9566295630615188, 0.8858682500839369, 1], [0.9614960581858876, 0.8816247775877105, 1], [0.47258744471967884, 0.2580020906487095, 1], [0.7581821747753655, 0.12970661177912624, 1], [0.1922818939415485, 0.5800013820737486, 1], [0.9937531092663561, 0.50046749650465, 1], [0.5176868690626143, 0.5231369531580075, 1], [0.798408868508208, 0.21825193764900785, 1], [0.24538476788962094, 0.011624306207803303, 1], [0.9653864267802522, 0.26538280031148653, 1], [0.7914743610758571, 0.6370910459553603, 1], [0.9782093454195158, 0.22535815375875867, 1], [0.6012466656508468, 0.36348254711056516, 1], [0.7294363835177021, 0.06565204181880091, 1], [0.8211586117427858, 0.5939868794256944, 1], [0.9605956583724057, 0.6454908363289914, 1], [0.1603962091436233, 0.08303344984263983, 1], [0.9281721455623533, 0.47134699576913097, 1], [0.508704406576266, 0.4854407446385436, 1], [0.8113947215892103, 0.3388565625680786, 1], [0.21276715580650218, 0.26523434030246296, 1], [0.9935158693570645, 0.3893481978513328, 1], [0.26503565828358056, 0.7558804259695867, 1], [0.8596729244529129, 0.2648000511198968, 1], [0.2916595257854946, 0.4615150962971326, 1], [0.8728609681807361, 0.5964395964564589, 1], [0.40654642331154667, 0.6066609102000513, 1], [0.7260221891888925, 0.010011047721346533, 1], [0.08544471115095054, 0.8774089310524329, 1], [0.958785728127713, 0.1698027356860845, 1], [0.44000556669107704, 0.8089124720778548, 1], [0.9705555463573466, 0.7566071441452125, 1], [0.3117735845513837, 0.2964752628178703, 1], [0.9151454284315136, 0.14304600703860326, 1], [0.453151659094711, 0.8663948556382663, 1], [0.8559666839057137, 0.2877431073233263, 1], [0.8879928348105282, 0.7127956513780495, 1], [0.8977275786672506, 0.4123856379540518, 1], [0.3830713649253503, 0.8640396827365648, 1], [0.8025751926688762, 0.2724478070207407, 1], [0.7519974638915492, 0.8909569535143954, 1], [0.9360985784775125, 0.7942370596973685, 1], [0.236603514201024, 0.3772932794935704, 1], [0.9835399312115355, 0.8555635793547497, 1], [0.2032705529266453, 0.8080468709081995, 1], [0.8953376078788613, 0.030823451862436624, 1], [0.006557437239696595, 0.34912580732108567, 1], [0.9398370205111403, 0.5382050069314778, 1], [0.08342963078934229, 0.18275834526116896, 1], [0.7500007629722526, 0.23215373016468432, 1], [0.13725739690690253, 0.4045254803040036, 1], [0.9215512250142126, 0.5781507209370556, 1], [0.79754000022526, 0.7051526584704815, 1], [0.9173943186644742, 0.14329109299358633, 1], [0.139089326374903, 0.6062948823357447, 1], [0.7705044657997674, 0.3104148310548064, 1], [0.47210473831038424, 0.8559692066630863, 1], [0.9547046001674476, 0.12244842117853783, 1], [0.6454901035167713, 0.95332910817832, 1], [0.931795603963539, 0.1449408210447013, 1], [0.9055554379037344, 0.8989770154272653, 1], [0.9033939226496454, 0.6750072423611285, 1], [0.3124056216607508, 0.013296884551639243, 1], [0.9706202386996597, 0.7544165522000169, 1], [0.4512536069431171, 0.45498046232386935, 1], [0.9418163477351426, 0.757019019712996, 1], [0.3382679540772694, 0.20047414753279746, 1], [0.9614372700072468, 0.6062513583878654, 1], [0.46643975384982606, 0.4119901935204705, 1], [0.9586649555700201, 0.6685081736155808, 1], [0.29141847980625235, 0.6656788499745873, 1], [0.8489825142388983, 0.43502645513637705, 1], [0.24564946917063202, 0.9232210809580058, 1], [0.9285834904133164, 0.3816468849015362, 1], [0.07778365378526675, 0.4531659826761204, 1], [0.9082671560801419, 0.2328187576895815, 1], [0.4097077629707784, 0.7974848272107903, 1], [0.916113965368667, 0.42014363500835206, 1], [0.6705660747799452, 0.8677854464379409, 1], [0.9826148328549941, 0.23973543167552525, 1], [0.12059260390367066, 0.45966666698021796, 1], [0.9349972920876851, 0.0038049014384752278, 1], [0.6019559833927142, 0.3248461730522172, 1], [0.8537402013566614, 0.4704243281504482, 1], [0.2893111802969901, 0.06630628657696469, 1], [0.8353459560700034, 0.49382364313112137, 1], [0.8259201687545011, 0.7598838626451474, 1], [0.8479108822551316, 0.0681045606002828, 1], [0.2721684609550369, 0.8635058814909023, 1], [0.7961389226527852, 0.33728858947901397, 1], [0.7113583615554396, 0.972581984226769, 1], [0.9092805939385062, 0.029483608327376176, 1], [0.19865244535238036, 0.6187149210892432, 1], [0.7762901531981664, 0.31594997997512964, 1], [0.34553142541128823, 0.4137425179220414, 1], [0.9925568017377652, 0.7372645234132206, 1], [0.6239390935497399, 0.1015248787478511, 1], [0.9783459035722393, 0.6983258295373881, 1], [0.2919471076511134, 0.7691017607908677, 1], [0.9527160538338639, 0.5312455589558756, 1], [0.4576580737462962, 0.7749911988178072, 1], [0.9818920658238444, 0.3745231448003731, 1], [0.7112822555128536, 0.32807821649056046, 1], [0.9073048758431246, 0.5765235750204595, 1], [0.8849899211345622, 0.9254822462806811, 1], [0.9469384493023947, 0.6915826143366871, 1], [0.24829512215023208, 0.09341626309179685, 1], [0.9763764012522872, 0.6095926275966626, 1], [0.43913055816856916, 0.8911363052662967, 1], [0.9766659946839237, 0.3251693290378532, 1], [0.6774611575558602, 0.5598142706528806, 1], [0.9365075046038777, 0.49826239402861905, 1], [0.07023171789258664, 0.7041738062498594, 1], [0.9246872160863977, 0.7036422957936439, 1], [0.4302812791354709, 0.6641125711649969, 1], [0.7277481292921149, 0.01894476806744405, 1], [0.1281574370996611, 0.12011398253092642, 1], [0.9654817179701296, 0.6024640058088464, 1], [0.1573289399650888, 0.7088664168847832, 1], [0.7664782729159674, 0.2970694259857397, 1], [0.37191667882905144, 0.9863944128661599, 1], [0.9343674665231352, 0.7292334763910778, 1], [0.4039484063248616, 0.724296924125215, 1], [0.9470564713335923, 0.4979860179052322, 1], [0.1966912093898303, 0.5432292408016265, 1], [0.9446110637495024, 0.5810310478369759, 1], [0.7483135345686882, 0.9652572621524822, 1], [0.950790769905533, 0.6864601425063975, 1], [0.7597983486031609, 0.3361074298966137, 1], [0.9782267125992019, 0.060289444399520664, 1], [0.2831904858215376, 0.9356289436109421, 1], [0.9737838262113163, 0.11825715437645878, 1], [0.041750568418972955, 0.7631067659331738, 1], [0.994110520865632, 0.679035417288684, 1], [0.30816178100636304, 0.0814767398357944, 1], [0.9249585449706594, 0.24498209488312683, 1], [0.6896556783816026, 0.43041976971754614, 1], [0.8267326864286791, 0.2673967809556852, 1], [0.3206967858253875, 0.36759421113292146, 1], [0.8919143681670044, 0.616031757699248, 1], [0.6472761822935242, 0.5643421210616058, 1], [0.9584682567059198, 0.19442798771600212, 1], [0.3727035888562129, 0.09359727612724433, 1], [0.9457708873989952, 0.4179139405444995, 1], [0.6686501940146523, 0.40945641669404786, 1], [0.9242211097086509, 0.6424683256857207, 1], [0.9155900831861407, 0.907985356175032, 1], [0.9155859505127509, 0.4779701774898506, 1], [0.8472148388130097, 0.9350797456951516, 1], [0.8283343027931692, 0.22993276736426538, 1], [0.7184432534331835, 0.47348382644668885, 1], [0.929765768032925, 0.3802206854844494, 1], [0.39466428810494536, 0.3662366152039358, 1], [0.7697692051678849, 0.21904464933398538, 1], [0.22429723620739273, 0.8065199412313473, 1], [0.9226767467133913, 0.5661494538183971, 1], [0.5643300333204854, 0.3458960883964912, 1], [0.8405408282256359, 0.08455899781480747, 1], [0.1187586232366995, 0.1389967460252307, 1], [0.9365364943567338, 0.5966928374053773, 1], [0.5456505542834358, 0.3841965974718322, 1], [0.8436206486104422, 0.3663438691477766, 1], [0.5554789850758399, 0.867238479894416, 1], [0.9845640218040332, 0.04513164214740506, 1], [0.3758758124228694, 0.4416971623767987, 1], [0.8319834661344844, 0.04727809260422544, 1], [0.5914824430689263, 0.21003984952562904, 1], [0.7310807736527208, 0.05323815134614762, 1], [0.698706993166715, 0.27754708434531195, 1], [0.9491113426130253, 0.3702473264549824, 1], [0.2885293505638886, 0.09791073849328813, 1], [0.7470679624337339, 0.03757981808675137, 1], [0.41143392074442675, 0.3697541870793183, 1], [0.9112165977240437, 0.35039311276216856, 1], [0.020539106829881137, 0.3353925006161549, 1], [0.9207678119876979, 0.44189488578455194, 1], [0.5325350237001607, 0.3005979764354545, 1], [0.9303645755966713, 0.32796202411345143, 1], [0.6376806525520066, 0.11489454437242697, 1], [0.8550369639012975, 0.14326763247879548, 1], [0.5924416586963743, 0.008356085825556803, 1], [0.9685514606570123, 0.31524248611769434, 1], [0.466645551603992, 0.2615594973137455, 1], [0.8135228369503333, 0.34591427418010423, 1], [0.47482019242088447, 0.9907116403137138, 1], [0.8517585442894624, 0.06863167213092669, 1], [0.43380677541758383, 0.34207283883268935, 1], [0.8218742391284283, 0.4312657064881372, 1], [0.4929105301173191, 0.5851381189081184, 1]]
#Y_train = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
T = 100
K = []

parameter1 = 0
parameter2 = 2
#kernel = 'Poly'
#X_train = [[1, 2, 1], [3, 4, 1], [5, 6, 1]]
#Y_train = [1, -1, 1]

NUM_INS = 1000
X_train = []
Y_train = []
sign = 1
i = 0
while i < NUM_INS:
    x = []
    for j in range(2):
        x.append(random.random() * 2)
    x.append(1)

    if 2*x[0] - x[1] > 2:
        if sign == 1:
            X_train.append(x)
            Y_train.append(1)
            sign = -1
            i += 1
    else:
        if sign == -1:
            X_train.append(x)
            Y_train.append(-1)
            sign = 1
            i += 1
            
X_train = array(X_train)
Y_train = array(Y_train)

if kernel == 'Linear':
	K = dot(X_train, X_train.transpose())
elif kernel == 'Poly':
        K = power(dot(X_train, X_train.transpose()) + parameter1, parameter2)
        #print K
else:
	print 'unknown kernel'

coefs = []
errors = []
samp_num = len(Y_train)
print samp_num

selection = [0]		# initialization: select the first sample point
selected = 0

coef = zeros((len(Y_train), 1))
coef[0] = float(Y_train[0]) / sqrt(K[0][0])

coefs = [coef]
preds = dot(K, coef)

errate = sum(Y_train * preds <= 0, axis=0)

for ii in range(1, samp_num):
	if ii % 10 == 0:
		print ii
	extension = selection + [ii]
	
	row_index = array(extension)
	col_index = array(extension)
	K_extend = K[row_index[:, None], col_index]
	#print 'K_extend = ', K_extend
	s, u = linalg.schur(K_extend)

        #if ii == 2:
 #               s = array([[93.22770,   -0.00000,    0.00000],[0.00000,    0.77230,    0.00000], [0.00000,    0.00000,   -0.00000]])
#                u = array([[0.24070, 0.88057, 0.40825], [0.52767, 0.23431, -0.81650], [0.81464, -0.41195, 0.40825]])

	s = diag(s)
	I = (s > tol)

	#print I
	#print u[:, I]
	#print power(s[I], -0.5)
	#print 's = ', s
	#print 'u = ', u
	#print 'u[:, I] = ', u[:, I]
	A = dot(u[:, I], diag(power(s[I], -0.5)))
        #print 'A = ', A
	
	#print 'selection ', selection
	#print 'extension ', extension
	#print 'shapeY ', shape(diag(Y_train[selection]))

	s_row_index = array(selection)
	s_col_index = array(extension)
	#print dot(diag(Y_train[selection]), K[s_row_index[:, None], s_col_index])
	restri = dot(dot(diag(Y_train[selection]), matrix(K[s_row_index[:, None], s_col_index])), A)
	#print 'restri =', restri
	co1 = dot(linalg.pinv(A), coef[extension])

        #print 'co1 = ', co1
	co2 = hit_n_run(co1, restri, T)
	co1 = hit_n_run(co2, restri, T)

	pred1 = dot(K[ii, extension], dot(A, co1))
	pred2 = dot(K[ii, extension], dot(A, co2))

	#print 'pred1 = ', pred1
	#print 'pred2 = ', pred2

	if pred1 * pred2 <= 0:
                selection = extension
                if Y_train[ii] * pred1 >= 0:
                        coef[extension] = dot(A, co1)
                else:
                        coef[extension] = dot(A, co2)


print selection
#print coef

#classifier = dot(power(X_train.transpose(), parameter2), coef)
#print 'classifier = ', classifier

classifier = dot(X_train.transpose(), coef)
print 'classifier = ', classifier

#results = dot(power(X_train, parameter2), classifier)
results = dot(X_train, classifier)

failed = 0
for idx in range(0,len(X_train)):
        if Y_train[idx] * results[idx] < 0:
                failed += 1

print 'failed = ', failed

'''
os.system('rm demo/bridge.xml-*')
os.system('./verifyta -t2 -f demo/bridge.xml demo/bridge.xml > /tmp/null 2>&1')

output = open('demo/bridge.xml-2.xtr')

for line in output.readlines():
	sys.stdout.write(line)
'''
