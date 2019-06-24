import tensorflow as tf
from tensorflow.keras import layers
import gym
import numpy as np
import logg as lg
import os
import tensorflow_probability as tfp

class Agent():
	def __init__(self, env, seed, batch_size, gamma, learning_rate, render = False):
		self.env = env
		self.ob_dim = env.observation_space.shape[0]
		self.is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
		if self.is_discrete:
			self.ac_dim = env.action_space.n
		else:
			self.ac_dim = env.action_space.shape[0]
			self.std = None
		self.max_p_length = env.spec.max_episode_steps
		self.seed = seed
		self.batch_size = batch_size
		self.gamma = gamma
		self.optimizer = tf.keras.optimizers.Adam(learning_rate)
		self.model = self.build_model()
		self.render = render

	#building policy NN
	def build_model(self):
		model = tf.keras.Sequential([
			layers.Dense(64, activation = 'tanh', input_shape = (self.ob_dim,)),
			layers.Dense(64, activation = 'tanh'),
			layers.Dense(self.ac_dim)])
		return model


	def sample_trajectories(self, env):
		total_batch_size = 0
		paths = []
		while True:
			path, steps = self.sample_trajectory(env)
			paths.append(path)
			total_batch_size += steps
			#restricts total batch size
			if self.batch_size <= total_batch_size:
				break
		return paths
	def policy_param(self, obs):
		if self.is_discrete:
			return self.model(obs)
		else:
			mean = self.model(obs)
			if self.std == None:
				# consider how to apply batch size here
				self.std = tf.Variable(tf.zeros(self.ac_dim),name = "std", dtype = tf.float32)
			return (mean, self.std)

	def sample_trajectory(self,env):
		steps = 0
		ob = env.reset()
		obs, acs, res = [], [], []
		while True:
			obs.append(ob)
			#notice that you need to make ob batch-like so the model can intake it.
			policy_param = self.policy_param(np.expand_dims(ob, axis = 0))
			ac = self.sample_action(policy_param)
			acs.append(ac)
			#render
			if self.render:
				self.env.render()
			ob, re, done, _ = env.step(ac)
			#sometimes, env returns nested list(ex : mountaincarcontinouous env)
			ob = np.squeeze(ob)
			res.append(re)
			steps += 1
			if done or steps >= self.max_p_length:
				break 
		path = {"acs": np.array(acs), "obs": np.array(obs), "res":np.array(res)}
		return path, steps

	def sample_action(self, policy_param):
		if self.is_discrete:
			logits = policy_param
			# remember tf.random.categorical intakes batch-shaped input and returns batch-shaped output
			sampled_ac = tf.random.categorical(logits, 1)
			#remember selecting element once is not enough because it is still a list form tensor. So do it twice
			sampled_ac = sampled_ac[0][0]
			#remember action here should not be a tensor, also remember 
			sampled_ac = sampled_ac.numpy()
		else:
			mean, std = policy_param
			eps = tf.random.normal([self.ac_dim])
			sampled_ac = mean + tf.exp(std) * eps
			sampled_ac = sampled_ac.numpy()

		return sampled_ac

	#train model
	def log_prob(self, policy_param, acs):
		if self.is_discrete:
			logits = policy_param
			log_prob = tf.keras.losses.sparse_categorical_crossentropy(y_true = acs, y_pred = logits, from_logits = True)
		else:
			mean, std = policy_param
			dist = tfp.distributions.MultivariateNormalDiag(mean, tf.math.exp(std))
			prob = tf.linalg.tensor_diag_part(dist.prob(acs))
			log_prob = -tf.math.log(prob)
		return log_prob

	def sum_rewards(self,res):
		def apply_discount(rew):
			#rew -> list
			temp = []
			res = []
			for i in range(len(rew)):
				temp.append(rew[i] * self.gamma**i)
			for i in range(len(temp)):
				res.append(sum(temp[i:]))
			return res
		q_n = []
		
		for i in range(len(res)):
			q_n.extend(apply_discount(res[i]))

		q_n = np.array(q_n).astype(np.float32)
		#normalize q_n
		q_n = (q_n - np.mean(q_n)) / np.std(q_n)

		return q_n



	def train_step(self, obs, acs, res):
		#remember "@tf.function" tensorizes elements so it is better to do preprocessing calculations beforehead for convenience.
		#Here, it means caculate qs beforehead
		@tf.function
		def train(obs, acs, qs):
			with tf.GradientTape() as gt:

				policy_param = self.policy_param(obs)
				log_prob = self.log_prob(policy_param, acs)
				objective = tf.reduce_mean(log_prob * qs)

			if self.is_discrete :
				training_vars = self.model.trainable_variables
			else :
				training_vars = self.model.trainable_variables + [self.std]

			grads = gt.gradient(objective, training_vars)
			self.optimizer.apply_gradients(zip(grads, training_vars))

		qs = self.sum_rewards(res)
		#tensorize variables so that there is no error under @tf.function
		obs = tf.cast(obs, dtype = obs.dtype)
		acs = tf.cast(acs, dtype = acs.dtype)
		qs = tf.cast(qs, dtype = qs.dtype)
		train(obs, acs, qs)

	def save_model(self, exp_name, seed, itr):
		model_dir = os.path.join("model", exp_name, str(seed))
		if not(os.path.exists(model_dir)):
			os.makedirs(model_dir)
		self.model.save(os.path.join(model_dir,'model_at_itr_{}'.format(itr)))


#train step
def train_PG(env_name, gamma, seed, n_iter, batch_size, lr, exp_name, n_experiment, render):
	env = gym.make(env_name)
	np.random.seed(seed)
	env.seed(seed)
	#try to let the agent independent from environment so that we can use the agent in other envs. And also think it is right concept
	#because we could introduce other agents with the same env.
	agent = Agent(env, seed, batch_size, gamma, lr, render)
	lg.initiate(exp_name, seed)
	for itr in range(n_iter):
		paths = agent.sample_trajectories(env)
		obs = np.concatenate([path["obs"] for path in paths])
		acs = np.concatenate([path["acs"] for path in paths])
		res = [path["res"] for path in paths]
		agent.train_step(obs, acs, res)
		if itr % 5 == 0:
			agent.save_model(exp_name, seed, itr)
		rewards = [res["res"].sum() for res in paths]
		os.system('clear')
		print("Experiment number {}".format(n_experiment + 1))
		print("iteration {}".format(itr + 1))
		lg.log("avg reward", np.mean(rewards))
		lg.log("std of reward", np.std(rewards))
		lg.log("max reward",np.max(rewards))
		lg.log("min reward", np.min(rewards))

	lg.finished(exp_name,['avg', 'std', 'max', 'min'] ,seed)

#main function which is running the entire process
def main():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("environment", type = str)
	parser.add_argument("--discount", "-d", type = float, default = 1.0)
	parser.add_argument("--n_experiment", "-n", type = int, default = 1)
	parser.add_argument("--n_iter", "-i", type = int, default = 100)
	parser.add_argument("--seed", "-s", type = int, default = 1)
	parser.add_argument("--batch", "-b", type = int, default = 1000)
	parser.add_argument("--learning_rate", "-lr", type = float, default = 5e-3)
	parser.add_argument("--exp_name", "-na", type = str, default = "hahaha")
	parser.add_argument("--render", action = 'store_true')


	arguments = parser.parse_args()
	for i in range(arguments.n_experiment):
		seed = arguments.seed + i
		train_PG(arguments.environment, arguments.discount, seed, arguments.n_iter, arguments.batch\
			, arguments.learning_rate, arguments.exp_name, i, arguments.render)




if __name__ == "__main__":
	main()