"""
A toy bitcoin project.
Adapted from imooc course: https://coding.imooc.com/lesson/214.html
"""
import json
import hashlib
import time
import uuid
import random
from urllib.parse import urlparse

import requests
from flask import Flask, request, jsonify

node_id = str(uuid.uuid4()).replace('-', '')
stop_mining = False

class Bitcoin:
    """A toy bitcoin"""

    def __init__(self):
        self.chain = []             # chain of blocks packed with transactions
        self.current_transactions = [] # unpacked transactions
        self.nodes = set()             # other nodes in the bitcoin network

        self.new_block(proof='100', previous_hash=1) # initial block

    def register_node(self, address):
        """register a new node in the bitcoin network"""
        parsed_url = urlparse(added)
        self.nodes.add(parsed_url.netloc)

    def valid_chain(self, chain) -> bool:
        """check if a chain is valid (not falsified)"""
        prev_block = chain[0]
        current_index = 1
        while current_index < len(chain):
            block = chain[current_index]
            if block['previous_hash'] != self.hash(prev_block):
                return False
            if not self.valid_proof(prev_block['proof'], block['proof']):
                return False
            prev_block = block 
            current_index += 1
        return True


    def resolve_conflict(self) -> bool:
        """find and extend the longest chain in the network"""
        neighbors = self.nodes
        max_length = len(self.chain)
        new_chain = None
        for node in neighbors:
            response = requests.get(f'http://{node}/chain')
            if response.status_code == 200:
                length = response.json()['length']
                chain = response.json()['chain']
                if length > max_length and self.valid_chain(chain):
                    max_length = length
                    new_chain = chain
        if new_chain:
            self.chain = new_chain
            return True
        return False

    def new_block(self, proof, previous_hash=None):
        block = {
                'index': len(self.chain)+1,
                'timestemp': time.time(),
                'transactions': self.current_transactions,
                'proof': proof,
                'previous_hash': previous_hash or self.hash(self.chain[-1])
                } 
        self.current_transactions = []
        self.chain.append(block)
        return block

    def new_transaction(self, sender, recipient, amount) -> int:
        """ make a new transaction.
        
        CAUTION: The transaction used here differ from that in bitcoin, 
        which adopts UTXO model.
        """
        tran = {
                'sender': sender,
                'recipient': recipient,
                'amount': amount,
                }
        # At this time the transaction should be broadcast and futher verified.
        self.current_transactions.append(tran)
        return self.last_block['index'] + 1

    @staticmethod
    def hash(block):
        block_string = json.dumps(block, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

    @property
    def last_block(self):
        return self.chain[-1]

    def proof_of_work(self, last_proof:int) -> int:
        """solve a diificult mathematical problem to show proof of work"""
        
        # if other node found a valid proof, this would be set to False 
        global stop_mining 
        proof = random.randint(1,1000)

        # until find a valid proof or other node has find one
        while (not self.valid_proof(last_proof, proof)) and (not stop_mining):
            proof += 1
        # print(proof)
        return proof

    def valid_proof(self, last_proof:int, proof:int) -> bool:
        guess = f'{last_proof}{proof}'
        guess_hash = hashlib.sha256(guess.encode()).hexdigest()
        # print(guess_hash)
        if guess_hash[:4] == '0000':
            return True
        else:
            return False


app = Flask(__name__)
bitcoin = Bitcoin()


@app.route('/transactions/new', methods=['POST'])
def new_transaction():
    values = request.get_json()
    index = bitcoin.new_transaction(values.get('sender'),
                                        values.get('recipient'),
                                        values.get('amount'))
    res = {'message': 'Transaction wil be added to block {0}'.format(index)}
    return jsonify(res), 201

@app.route('/mine', methods=['GET'])
def mine():
    global stop_mining
    last_block = bitcoin.last_block
    last_proof = last_block['proof']
    proof = bitcoin.proof_of_work(last_block)
    if stop_mining:
        stop_mining = False
        return jsonify({'message': 'Minning stopped!'}), 201
    all_nodes =  bitcoin.nodes()
    for n in all_nodes:
        r = requests.post(n+'/stop_mining', data={'proof': proof})
    bitcoin.new_transaction("0",node_id, 12.5)
    block = bitcoin.new_block(proof, None)
    res ={
            'message': 'New block created!',
            'index': block['index'],
            'transactions': block['transactions'],
            'proof': block['proof'],
            'previous_hash': block['previous_hash']
            }
    return jsonify(res), 201

@app.route('/stop_minning', methods=['POST'])
def stop_mining_and_verify():
    last_block = bitcoin.last_block
    last_proof = last_block['proof']
    proof = request.json()['proof']
    if bitcoin.valid_proof(last_block, proof):
        stop_mining = True
        res = {'message': 'Stopped and verified your proof.'}
        return jsonify(res), 201
    else:
        res = {'message': 'Your proof is not valid!'}
        return jsonify(res), 201

@app.route('/chain', methods=['GET'])
def full_chain():
    res = {
            'chain': bitcoin.chain,
            'length': len(bitcoin.chain)
            }
    return jsonify(res), 200

@app.route('/nodes/register', methods=['POST'])
def register_nodes():
    values = request.get_json()
    nodes = values.get('nodes')
    if nodes is None:
        return "Error: please provide a valid node list"

    for node in nodes:
        bitcoin.register_node(node)

    res = {
            'message': 'New nodes registered',
            'total_nodes': list(bitcoin.chain)
            }
    return jsonify(res), 201

@app.route('/nodes/resolve', methods=['GET'])
def consensus():
    replaced = bitcoin.resolve_conflict()
    if replaced:
        res = {
                'message': 'Our chain has been replaced!',
                'new_chain': bitcoin.chain
                }
    else:
        res = {
                'message': 'Our chian has been authoritative!',
                'new_chain': bitcoin.chain
                }
    return jsonify(res), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
