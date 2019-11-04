
# 텐서플로우를 켠다. numpy도 켠다.
import tensorflow as tf
import numpy as np

# 사용할 데이터 셋을 만든다.(*placeholder는 파라미터 용도, Variable이 allocation)
print("FLAGS is",tf.app.flags.FLAGS)
n_dim = 39 # 소리파일에서 추출한 데이터의 차원 수(우리는 80차 벡터를 씀)
n_classes = 5 # 결과값으로 분류할 가짓 수(우리는 총 2가지 음향을 분류함)
n_hid1 = 300 # hidden layer 1의 차원 수
n_hid2 = 200 # hidden layer 2의 차원 수
n_hid3 = 100 # hidden layer 3의 차원 수

training_epochs = 50000 # 학습 횟수
learning_rate = 0.01 # 학습 비율
sd = 1 / np.sqrt(n_dim) # standard deviation 표준편차(표본표준편차라 1/root(n))

# 입력 데이터 파라미터와 정답 데이터 파라미터를 생성한다.
X = tf.placeholder(tf.float32, [None, n_dim])
Y = tf.placeholder(tf.float32, [None, n_classes])

# 1차 히든 레이어(원소까지 랜덤인 배열을 생성)
W_1 = tf.Variable(tf.random_normal([n_dim, n_hid1], mean=0, stddev=sd), name="w1")
b_1 = tf.Variable(tf.random_normal([n_hid1], mean=0, stddev=sd), name="b1")
#h_1 = tf.nn.sigmoid(tf.matmul(X, W_1) + b_1) # 1차 히든레이어는 '시그모이드' 함수를 쓴다.
h_1 = tf.nn.relu(tf.matmul(X, W_1) + b_1) # 1차 히든레이어는 'Relu' 함수를 쓴다.

# 2차 히든 레이어
W_2 = tf.Variable(tf.random_normal([n_hid1, n_hid2], mean=0, stddev=sd), name="w2")
b_2 = tf.Variable(tf.random_normal([n_hid2], mean=0, stddev=sd), name="b2")
h_2 = tf.nn.tanh(tf.matmul(h_1, W_2) + b_2) # 2차 히든레이어는 '하이퍼볼릭탄젠트' 함수를 쓴다.

# 3차 히든 레이어
W_3 = tf.Variable(tf.random_normal([n_hid2, n_hid3], mean=0, stddev=sd), name="w3")
b_3 = tf.Variable(tf.random_normal([n_hid3], mean=0, stddev=sd), name="b3")
h_3 = tf.nn.relu(tf.matmul(h_2, W_3) + b_3) # 3차 히든레이어는 'relu' 함수를 쓴다.

# 드롭아웃 과정 추가
keep_prob = tf.placeholder(tf.float32)
h_3_drop = tf.nn.dropout(h_3, keep_prob)

# 최종 evidence 레이어(?? 이거 뭐라고 불러야하지)
W = tf.Variable(tf.random_normal([n_hid3, n_classes], mean=0, stddev=sd), name="w")
b = tf.Variable(tf.random_normal([n_classes], mean=0, stddev=sd), name="b")
y_ = tf.nn.softmax(tf.matmul(h_3_drop, W) + b) # 소프트맥스 처리한 것 하나 이렇게 총 두개를 만들어둔다.(왜??)
# 소프트맥스 함수 : 확률화 함수( 전체중 비율이 어느정도인지 매겨줌)
# 각 y의 원소값을 sum(y)로 나눠준다

# 설명 1) 이런 식으로 각 레이어들의 W,b,h가 차례차례 연쇄적으로 연산된다.
#      2) 기계가 추측한 결과는 y_에, 실제 정답은 Y에 저장된다.

# '교차 엔트로피' 방식으로 오차를 측정한다.(cross-entropy cost function 사용)
#cross_entropy = -tf.reduce_sum(Y*tf.log(y_))
#cross_entropy = -tf.reduce_sum(Y*tf.log(tf.clip_by_value(y_, 1e-10, 1.0)))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(y_), reduction_indices=[1])) # 웹사이트의 CNN

# 오차를 줄이는 방향으로 학습한다.(여기서 두 가지 학습 방법이 있다.)
# 1) 책의 CNN
#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 2) 웹사이트의 CNN
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

#학습이 완료되면 정답률을 체크한다.
correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 트레이닝된 데이터 저장
saver = tf.train.Saver() # 트레이닝된 데이터 저장

# -> 함수 작성 끝! 이제 학습을 돌려보면 된다.

# 세션을 켜고, 초기화한다.
with tf.Session() as sess: # 이러면 끝나고 세션을 자동으로 닫아준다.
    sess.run(tf.initialize_all_variables())
    data = np.load("mytotaldata.npz")
    tc = data['X']
    td = data['Y']
	# 지정된 횟수만큼 학습한다.
    for epoch in range(training_epochs):
        sess.run(train_step, feed_dict={X: tc, Y: td, keep_prob: 0.5})
        # 100번마다 정확도가 출력된다.
        if epoch%100 == 0 :
            train_accuracy = sess.run(accuracy, feed_dict={X: tc, Y: td, keep_prob: 1.0})
            print("step %d, training accuracy %g"%(epoch, train_accuracy))
            if(train_accuracy > 0.90) :
                print("step %d, training success!"%epoch)
                break
    # 평가
    print(sess.run(accuracy, feed_dict={X: tc, Y: td, keep_prob: 1.0}))
    save_path = saver.save(sess,"./mytotalmodel.ckpt")
    print("The model is saved in file as, : ", save_path)