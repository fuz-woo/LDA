from numpy import *

import cPickle as pickle
corpus = pickle.load(open("corpus-for-test.dump"))

class LDA:
    def __init__(self, corpus, K=20, alpha=0.05, beta=0.01, n_iter=300):
        self.corpus = corpus
        self.M = len(self.corpus)
        self.NM = sum(map(len,self.corpus))
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.n_iter = n_iter
        
        self.word_topics = [random.randint(0,self.K,size=l) for l in map(len,self.corpus)]
        self.wdic = list(set([w for doc in self.corpus for w in doc]))
        self.wsize = len(self.wdic)

        self.n_burn_in = 100

        print "corpus size:",self.M
        print "word count: ",self.wsize

        self.n_z = [{w:0 for w in self.wdic} for k in range(self.K)]
        self.n_z_sum = [0 for k in range(self.K)]
        self.n_m = [[0 for k in range(self.K)] for m in range(self.M)]
        self.n_m_sum = [0 for m in range(self.M)]

        for m in range(self.M):
            doc = self.corpus[m]
            for i in range(len(doc)):
                w = self.corpus[m][i]
                z = self.word_topics[m][i]
                self.n_m[m][z] += 1
                self.n_z[z][w] += 1
                self.n_m_sum[m] += 1
                self.n_z_sum[z] += 1

        self.phi = [{w:0 for w in self.wdic} for k in range(self.K)]
        self.theta = [[0 for k in range(self.K)] for m in range(self.M)]
        self.nstats = 0

        self.perplexities = []


    def run(self):
        for n in range(self.n_iter):
            print "iteration ",n
            self.sample_routine()
#if n > self.n_burn_in:
            self.update_params()
            pplex = self.perplexity()
            self.perplexities.append(pplex)
            print "perplexity: ", pplex

    def sample_routine(self):
        for m in range(self.M):
            doc = self.corpus[m]
            N = len(doc)
            for i in range(N):
                w = self.corpus[m][i]
                z = self.word_topics[m][i]
                
                self.n_m[m][z] -= 1
                self.n_z[z][w] -= 1
                self.n_m_sum[m] -= 1
                self.n_z_sum[z] -= 1

                newz = self.sample(m,i)
                self.word_topics[m][i] = newz

                self.n_m[m][newz] += 1
                self.n_z[newz][w] += 1
                self.n_m_sum[m] += 1
                self.n_z_sum[newz] += 1


    def sample(self,m,i):
        w = self.corpus[m][i]
        z = self.word_topics[m][i]

        msum = (self.n_m_sum[m] + self.K * self.alpha)

        p = [(self.n_m[m][k] + self.alpha) * (self.n_z[k][w] + self.beta) / (msum * (self.n_z_sum[k] + self.wsize * self.beta)) for k in range(self.K)]
        _psum = sum(p)
        p = [pi/_psum for pi in p]

        newz = random.choice(range(self.K),p=p)
        return newz
        
        
    def update_params(self):
        self.phi = [{w:0 for w in self.wdic} for k in range(self.K)]
        self.theta = [[0 for k in range(self.K)] for m in range(self.M)]
        for m in range(self.M):
            doc = self.corpus[m]
            for k in range(self.K):
                self.theta[m][k] = (self.n_m[m][k] + self.alpha) / (self.n_m_sum[m] + self.K * self.alpha)

        for k in range(self.K):
            for w in self.wdic:
                self.phi[k][w] = (self.n_z[k][w] + self.beta) / (self.n_z_sum[k] + self.wsize * self.beta)
        self.nstats += 1

    def show_topics(self,num_topics=20,num_words=20):
        self.topics = [sorted(phi.items(),key=lambda (w,c):c, reverse=True) for phi in self.phi]
        self.topics = [[(w,c/self.nstats) for w,c in topic][:num_words] for topic in self.topics][:num_topics]
        return self.topics

    def get_phi(self):
        phis = [sorted(phi.items(),key=lambda (w,c):c, reverse=True) for phi in self.phi]
        phis = [dict([(w,c/self.nstats) for w,c in phi]) for phi in phis]
        return phis

    def get_theta(self):
        thetas = [[kc/self.nstats for kc in doc] for doc in self.theta]
        return thetas
        

    def perplexity(self):
        phis = self.phi
        thetas = self.theta
        pplex = 0.
        for m in range(self.M):
            doc = self.corpus[m]
            for n in range(len(doc)):
                t = self.corpus[m][n]
                s = 0.
                for k in range(self.K):
                    s += phis[k][t]*thetas[m][k]
                pplex += log(s) / self.NM
        return exp(0. - pplex)

lda = LDA(corpus)
