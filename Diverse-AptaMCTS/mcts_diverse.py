## My adapted version include a constant 11 nt ms2 region

import time
import math
import random
import joblib
import RNA
from utils import *

ENV = None


class TreeNode:
    def __init__(self, state, parent):
        self.state = state
        self.is_terminal = state.is_terminal()
        self.is_fully_expanded = self.is_terminal
        self.parent = parent
        self.num_visits = 0
        self.total_reward = 0
        self.children = {}


def random_policy(state):
    while not state.is_terminal():
        try:
            action = random.choice(state.get_possible_actions())
        except IndexError:
            raise Exception("No possible action in this state {}".format(str(state)))
        state = state.take_action(action)
    return state.get_reward()


class MCTS:
    def __init__(self, time_limit=None, iteration_limits=None,
                 exploration_constant=1 / math.sqrt(2), rollout_policy=random_policy,
                 commit_temperature=0.0):
                 # added temperature feature to allow stochastic commit of root action (0 == original argmax)
                 # also can change exploration constant to control exploration/exploitation tradeoff in UCB1 selection

        if time_limit != None:
            if iteration_limits != None:
                raise ValueError("Cannot have both a time limit and an iteration limit!")

            # time taken for each MCTS search in milliseconds
            self.time_limit = time_limit
            self.limit_type = "time"
        else:
            if iteration_limits == None:
                raise ValueError("Must have either a time limit or an iteration limit!")

            # number of iteractions of the search
            if iteration_limits < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.search_limit = iteration_limits
            self.limit_type = 'iteration'

        self.exploration_constant = exploration_constant
        # Temperature for the per-position commit. 0.0 == original argmax
        # (pure exploitation); > 0 samples the committed letter from the root
        # children weighted by visit_count**(1/T), so independent runs commit
        # different prefixes and explore different motifs.
        self.commit_temperature = commit_temperature #!!!

        self.rollout = rollout_policy
        self.candidates = []

    def get_candidates(self):
        return self.candidates

    def get_root_childs(self):
        return self.root.get_children()

    def search(self, initial_state):
        self.root = TreeNode(initial_state, None)  # parent -> none

        if self.limit_type == "time":
            time_limit = time.time() + self.time_limit / 1000
            while time.time() < time_limit:
                self.execute_round()
        else:
            for i in range(self.search_limit):
                self.execute_round()

        return self._commit_action()

    def execute_round(self):

        # selection
        node = self.select_node(self.root)

        # expansion & simulation
        reward, candidate_data = self.rollout(node.state)
        self.candidates.append(candidate_data)

        # backpropagation
        self.backpropagation(node, reward)

    def select_node(self, node):
        while not node.is_terminal:
            if node.is_fully_expanded:
                node = self.get_best_child(node, self.exploration_constant)
            else:
                return self.expand(node)
        return node

    def expand(self, node):
        actions = node.state.get_possible_actions()
        for action in actions:
            if action not in node.children:
                next_state = node.state.take_action(action)
                parent_node = node
                new_node = TreeNode(next_state, parent_node)
                node.children[action] = new_node
                if len(actions) == len(node.children):
                    node.is_fully_expanded = True
                return new_node
        raise Exception("Non reachable error")

    def backpropagation(self, node, reward):
        while node is not None:
            node.num_visits += 1
            node.total_reward += reward
            node = node.parent

    def get_best_child(self, node, exploration_value):
        best_value = float("-inf")
        best_nodes = []
        for child in node.children.values():

            node_value = child.total_reward / child.num_visits
            node_value += exploration_value * math.sqrt(2 * math.log(node.num_visits) / child.num_visits)

            if node_value > best_value:
                best_value = node_value
                best_nodes = [child]
            elif node_value == best_value:
                best_nodes.append(child)

        return random.choice(best_nodes)

    def get_action(self, root, best_child):
        for action, node in root.children.items():
            if node is best_child:
                return action

    def _commit_action(self):
        """Choose which root action to commit this round.

        commit_temperature == 0 -> original behaviour: argmax mean reward
        (exploration weight 0). commit_temperature > 0 -> sample among root
        children with probability proportional to num_visits**(1/T). Higher T
        => flatter distribution => more diverse committed prefixes across runs.
        """
        children = list(self.root.children.items())  # (action, node)
        T = self.commit_temperature
        if T is None or T <= 0 or len(children) <= 1:
            best_child = self.get_best_child(self.root, 0)
            return self.get_action(self.root, best_child)

        weights = [node.num_visits ** (1.0 / T) for _, node in children]
        total = sum(weights)
        if total <= 0:
            best_child = self.get_best_child(self.root, 0)
            return self.get_action(self.root, best_child)
        action, _ = random.choices(children, weights=weights, k=1)[0]
        return action


class Action:
    def __init__(self, cur_bp, next_letter):
        self.cur_bp = cur_bp
        self.next_letter = next_letter

    def get_next_letter(self):
        return self.next_letter

    def __str__(self):
        return str((self.cur_bp, self.next_letter))

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.cur_bp == other.cur_bp and self.next_letter == other.next_letter

    def __hash__(self):
        return hash((self.cur_bp, self.next_letter))


class Environment:
    def __init__(self, model):
        self.model = model  # scikit-learn random forest classifier
        self.p_mer = 3
        self.r_mer = 4
        self.p_iCTF = improvedCTF(letters=["A", "B", "C", "D", "E", "F", "G"], length=self.p_mer)
        self.r_iCTF = improvedCTF(letters=["A", "C", "G", "U"], length=self.r_mer)
        self.reduced_p_dict = get_reduced_protein_letter_defaultdict()
        self.fwd_seq = None
        self.bwd_seq = None

    def init_target_protein(self, seq):
        self.target_p_seq = seq
        self.px = self.encoder_p(seq)

    def init_spe_proteins(self, seqs):
        self.spe_p_seqs = seqs
        self.spe_px_list = [self.encoder_p(seq) for seq in seqs]

    def set_forward_sequence(self, fwd_seq):
        self.fwd_seq = fwd_seq

    def set_backward_sequence(self, bwd_seq):
        self.bwd_seq = bwd_seq

    def encoder_p(self, seq):
        seq_len = len(seq)
        pf_dict = self.p_iCTF.get_feature_dict()
        rpseq = []
        for p in seq:
            rpseq.append(self.reduced_p_dict[p])

        temp_pseq = ""
        for p in rpseq:
            temp_pseq += p
        seq = temp_pseq

        for mer in range(1, self.p_mer + 1):
            for i in range(0, len(seq) - mer):
                pattern = seq[i:i + mer]
                try:
                    pf_dict[pattern] += 1
                except:
                    continue
        pf = np.array(list(pf_dict.values()))
        pf = pf / seq_len
        return pf

    def encoder_r(self, seq):
        seq_len = len(seq)
        rf_dict = self.r_iCTF.get_feature_dict()
        for mer in range(1, self.r_mer + 1):
            for i in range(0, len(seq) - mer):
                pattern = seq[i:i + mer]
                try:
                    rf_dict[pattern] += 1
                except:
                    continue

        rf = np.array(list(rf_dict.values()))
        rf = rf / seq_len
        return rf

    def get_reward(self, aptamer_sequence):

        if self.fwd_seq is not None:
            aptamer_sequence = self.fwd_seq + aptamer_sequence

        if self.bwd_seq is not None:
            aptamer_sequence = aptamer_sequence + self.bwd_seq

        rx = self.encoder_r(aptamer_sequence)
        rx = list(rx)
        x = np.array([list(self.px) + rx])

        spe_x = []
        for spe_px in self.spe_px_list:
            spe_x.append(list(spe_px) + rx)

        if len(spe_x) > 0:
            spe_x = np.array(spe_x)
            y_pred_prob = self.model.predict_proba(x)[0][1]
            y_pred_prob_spe = self.model.predict_proba(spe_x)[:, 1]
            return y_pred_prob, y_pred_prob_spe, aptamer_sequence
        else:
            y_pred_prob = self.model.predict_proba(x)[0][1]
            return y_pred_prob, 0, aptamer_sequence


def act8_aptamer_to_string(best_aptamer):
    reordered_aptamer = ""
    for apt in best_aptamer:
        if apt in "acgu":
            reordered_aptamer = reordered_aptamer + apt
        elif apt in "ACGU":
            reordered_aptamer = apt + reordered_aptamer
        else:
            continue
    return reordered_aptamer.upper()

def assemble_with_const(build_order_str, block1_len, const_region):
    """Decode a two-block build-order string into a full core with the
    constant region spliced between the two variable blocks.
 
    Each block is decoded INDEPENDENTLY so that the directional (upper/lower)
    encoding of act8 cannot reorder bases across the fixed constant boundary.
 
    Args:
        build_order_str (str): the raw state string (block1_len + block2_len chars)
        block1_len (int): number of variable positions in the first (5') block
        const_region (str): fixed constant nucleotides inserted between blocks
 
    Returns:
        str: block1 (decoded) + const_region + block2 (decoded)
    """
    b1 = act8_aptamer_to_string(build_order_str[:block1_len])
    b2 = act8_aptamer_to_string(build_order_str[block1_len:])
    return b1 + const_region + b2

class AptamerStates:
    def __init__(self, bp=27, letters=["A", "C", "G", "U", "a", "c", "g", "u"],
                 current_aptamer="", block1_len=None, const_region="", block2_len=None):
        self.n_letters = len(letters)
        self.aptamer = current_aptamer
        self.actions = letters
 
        # Two-block-with-constant configuration.
        # If block lengths are not supplied, fall back to a single block of
        # length `bp` and an empty constant (== original behaviour).
        if block1_len is None and block2_len is None:
            self.block1_len = bp
            self.block2_len = 0
        else:
            self.block1_len = block1_len if block1_len is not None else 0
            self.block2_len = block2_len if block2_len is not None else 0
 
        self.const_region = const_region
        # bp is the total number of VARIABLE positions the search fills.
        # The constant is NOT searched, so it is excluded from this count.
        self.bp = self.block1_len + self.block2_len

    def get_possible_actions(self):
        """
        State is sequene of the Aptamer

        - aptamer has 4 letters (DNA case ACGT, RNA case ACGU)
        - that means possible actions are only 4 actions
        - But! in this version, we choose 8 actions which is multiplied left or right directions

        """

        possible_actions = [Action(len(self.aptamer), nl) for nl in self.actions]
        return possible_actions

    def take_action(self, action):
        """
        Select possible action and update states

        - In this case, aptamer is sequence of 4 letters x 2 direction(left, right) of concatenation
        - so the next state(aptamer string) just add action(next letter) to the current state

        """
        new_state = deepcopy(self)
        new_state.aptamer += action.get_next_letter()
        return new_state

    def is_terminal(self):
        """ Check the end of state
        - we already specified target length of aptamer
        - if length of aptamer(state) is same with the length then terminal
        """
        if len(self.aptamer) == self.bp:
            return True
        else:
            return False

    def get_reward(self):
        # Splice the fixed constant between the two decoded variable blocks.
        full_core = assemble_with_const(self.aptamer, self.block1_len, self.const_region)
        reward, spe_rewards, full_aptamer = ENV.get_reward(full_core)
        ss, mfe = RNA.fold(full_aptamer)
        candidate_data = (reward, full_aptamer, ss, mfe)
 
        # apply specificity rewards to the target bind reward
        specificity_coef = 1.0
 
        total_reward = reward - np.mean(spe_rewards) * specificity_coef
 
        return total_reward, candidate_data
    
def _kmer_set(seq, k=4):
    return {seq[i:i + k] for i in range(len(seq) - k + 1)}


def _similarity(a, b, k=4):
    """Jaccard similarity on k-mer sets (0 = disjoint, 1 = identical k-mer content)."""
    sa, sb = _kmer_set(a, k), _kmer_set(b, k)
    if not sa or not sb:
        return 1.0 if a == b else 0.0
    return len(sa & sb) / len(sa | sb)


def mmr_select(candidates, top_k, lam=1.0, k=4):
    """Maximal Marginal Relevance selection over scored candidates.

    Greedily builds a set that trades off score against novelty relative to
    what has already been picked:

        value = lam * score - (1 - lam) * max_similarity_to_selected

    lam == 1.0 reproduces plain top-k by score (no diversity pressure).
    Lower lam => more diverse pool (motif families spread out rather than
    returning many near-variants of the single best sequence).

    Args:
        candidates (list): (score, seq, ss, mfe) tuples, assumed sorted desc by score
        top_k (int): number to return
        lam (float): score/diversity trade-off in [0, 1]
        k (int): k-mer size for the similarity measure
    """
    if lam >= 1.0:
        return candidates[:top_k]

    selected = []
    pool = list(candidates)
    while pool and len(selected) < top_k:
        best_i, best_val = 0, float("-inf")
        for i, cand in enumerate(pool):
            seq = cand[1]
            sim = max((_similarity(seq, s[1], k) for s in selected), default=0.0)
            val = lam * cand[0] - (1.0 - lam) * sim
            if val > best_val:
                best_val, best_i = val, i
        selected.append(pool.pop(best_i))
    return selected


class AptaMCTS:
    def __init__(self, score_function_path):
        self.sf_path = score_function_path  # pre-trained rf-model path
        self.__load_score_function()

    def __load_score_function(self):
        self.score_function = joblib.load(self.sf_path)
        self.score_function.set_params(n_jobs=1)
        print("")
        print("> Load RF model with {} trees".format(len(self.score_function.estimators_)))
        print("- from : {}".format(self.sf_path))
        print("")

    def __redundancy_removal(self, candidates):
        ss_dict = defaultdict(lambda: {"score": -1, "seq": None, "mfe": None})
        for score, seq, ss, mfe in candidates:
            if ss_dict[ss]["score"] < score:
                ss_dict[ss]["score"] = score
                ss_dict[ss]["seq"] = seq
                ss_dict[ss]["mfe"] = mfe

        _candidates = []
        for ss, vals in ss_dict.items():
            score = vals["score"]
            seq = vals["seq"]
            mfe = vals["mfe"]
            _candidates.append((score, seq, ss, mfe))

        return _candidates

    def sampling(self, target_pseq, target_bp, top_k, n_iter, p_spes, fwd_seq, bwd_seq,
    block1_len=None, const_region="", block2_len=None,
    restarts=1, exploration_constant=1 / math.sqrt(2), commit_temperature=0.0,
    mmr_lambda=1.0):
    # new args are block1_len, const_region, block2_len to support two-block-with-constant configuration
    # if no new args are provided, it will default to the original single-block behaviour
    #
    # Diversity controls (all default to original behaviour):
    #   restarts             - run the greedy build this many times, pooling all
    #                          rollout candidates before selection. Only useful
    #                          when commit_temperature > 0 (else every restart is
    #                          near-identical).
    #   exploration_constant - UCB1 exploration weight (was hardcoded 1/sqrt(2)).
    #   commit_temperature   - stochastic per-position commit (0 == argmax).
    #   mmr_lambda           - diversity-aware final selection (1.0 == plain top-k).
        global ENV
        ps_names, ps_seqs = p_spes
        aptamer_letters = ["A", "C", "G", "U", "a", "c", "g", "u"]  # directional ribonucleotide bases
        ENV = Environment(self.score_function)  # global parameter (as a score function) initialization
        ENV.init_target_protein(target_pseq)
        ENV.init_spe_proteins(ps_seqs)

        if fwd_seq is not None:
            ENV.set_forward_sequence(fwd_seq)

        if bwd_seq is not None:
            ENV.set_backward_sequence(bwd_seq)

        if block1_len is None and block2_len is None:
            b1, b2 = target_bp, 0
        else:
            b1 = block1_len if block1_len is not None else 0
            b2 = block2_len if block2_len is not None else 0
 
        total_var = b1 + b2  # total number of variable positions to commit
 
        print("> Search config: block1={} nt | const={} nt ('{}') | block2={} nt | total variable={} nt"
              .format(b1, len(const_region), const_region, b2, total_var))

        print("> Diversity config: restarts={} | exploration_constant={:.4f} | "
              "commit_temperature={} | mmr_lambda={}"
              .format(restarts, exploration_constant, commit_temperature, mmr_lambda))

        candidates = []

        # Greedy outer loop: commit one variable letter per inner MCTS run.
        # This naturally runs b1 rounds for block 1, then b2 rounds for block 2.
        # The constant is auto-spliced at scoring time (see AptamerStates.get_reward),
        # so every full-length rollout scores block1 + const + block2.
        #
        # Outer restart loop pools rollout candidates across independent builds;
        # with commit_temperature > 0 each restart commits a different prefix and
        # therefore explores a different motif.
        for r in range(restarts):
            best_aptamer = ""
            while len(best_aptamer) < total_var:
                initial_state = AptamerStates(current_aptamer=best_aptamer,
                                              letters=aptamer_letters,
                                              block1_len=b1,
                                              const_region=const_region,
                                              block2_len=b2)

                mcts = MCTS(time_limit=None, iteration_limits=n_iter,
                            exploration_constant=exploration_constant,
                            commit_temperature=commit_temperature)
                action = mcts.search(initial_state=initial_state)
                next_letter = action.get_next_letter()
                best_aptamer += next_letter
                candidates += mcts.get_candidates()
                print(next_letter, end="")
            print("  [restart {}/{}] -> {}".format(r + 1, restarts, best_aptamer))

        _candidates = self.__redundancy_removal(candidates)
        print("- Total number of candidates : {} (original {})".format(len(_candidates), len(candidates)))
        _candidates = list(sorted(_candidates, key=lambda x: -x[0]))
        _candidates = mmr_select(_candidates, top_k, lam=mmr_lambda)

        return _candidates