from pymongo import MongoClient
from pymongo.errors import BulkWriteError
import numpy as np

UPDATED = True


class DB:
    """
        DB interface for markov decision process
        for CRUD functions of state and value
    """

    def __init__(self, db_name, top_exceed=1000000):
        self.client = MongoClient()
        self.db = self.client[db_name]
        self.col_states = self.db.states
        self.top_exceed = top_exceed
        self.states = None
        self.values = None
        self.updated_tags = None

    def load_data(self):
        """ load data from db"""
        datas = self.db.states.find({}).sort('_id', -1).limit(self.top_exceed)
        states = []
        values = []
        updated = []
        for data in datas:
            if data['value'] != 0.0:
                states.append(data['state'])
                values.append(data['value'])
                updated.append(False)
        self.states = np.array([state for state in states])
        self.values = np.array([value for value in values])
        self.updated_tags = np.array([tag for tag in updated])
        return None

    def bulk_save(self):
        """
            bulk save data
            TODO: bulk save is too heavy, add a updated tag to specify which to save
        """
        bulk = self.col_states.initialize_unordered_bulk_op()
        indexes = np.argwhere(self.updated_tags == True)
        for index in indexes:
            bulk.find({'state': self.states[index[0]].tolist()}).upsert().update({
                '$set': {
                    'value': self.values[index].tolist()[0]
                }
            })
            self.updated_tags[index] = False
        try:
            bulk.execute()
        except BulkWriteError as bwe:
            print bwe.details

    def push(self, state, value):
        """push a data to in memory data"""
        self.states = np.append(self.states, [state], axis=0)
        self.values = np.append(self.values, [value], axis=0)
        self.updated_tags = np.append(self.updated_tags, [UPDATED], axis=0)

    def pop(self):
        """
            pop out a data incase memory explosion
        """
        state = np.delete(self.states, (0), axis=0)
        value = np.delete(self.values, (0), axis=0)
        updated = np.delete(self.updated_tags, (0), axis=0)
        return state, value, updated

    def find_state_in_memory(self, state):
        """
            find data in memory
        """
        index = np.argwhere(np.all(self.states == state, axis=(2, 1)))
        if index.shape[0] == 0:
            return None
        else:
            return index[0][0]

    def find_state_in_db(self, state):
        """
            find state
            type(state) is np.ndarray
        """
        result = self.col_states.find_one({'state': state.tolist()})
        return result

    def find_value(self, state, default_value):
        """
            find state, first search in memory then in database
        """
        index = self.find_state_in_memory(state)
        value = None
        in_memory = None
        if isinstance(index, int):
            # value in memory
            value = self.values[index]
            in_memory = True
        else:
            # value not in memory
            result = self.find_state_in_db(state)
            if result:
                value = result['value']
            else:
                value = default_value
            in_memory = False
        return index, value, in_memory

    def store_state_in_memory(self, state, value):
        """
            store state in memory
        """
        self.push(state, value)

    def store_state_in_db(self, state, value):
        """
            store value
        """
        result = self.col_states.update_one({
            'state': state.tolist()
        }, {
            '$set': {
                'value': value
            }
        }, upsert=True)
        return result

    def store_state(self, state, value, default_value):
        """
            store state in memory first
            if memory is full, pop out the first record, then store the poped record to database
        """
        in_memory_number = self.values.shape[0]
        index,  previous_value,  in_memory = self.find_value(state, default_value)
        if previous_value == value:
            return
        elif in_memory:
            self.update_state_in_memory(index, value)
        else:
            self.store_state_in_memory(state, value)
            if in_memory_number <= self.top_exceed:
                # do nothing
                return
            else:
                poped_state, poped_value, updated = self.pop()
                if updated:
                    self.store_state_in_db(poped_state, poped_value)

    def update_state_in_memory(self, index, value):
        self.values[index] = value
        self.updated_tags[index] = True 

    def find_values(self, states, default_value):
        """get values"""
        v = []
        for state in states:
            result = self.find_value(state, default_value)
            if result and 'value' in result:
                value = result['value']
                v.append(value)
            else:
                v.append(default_value)
        return np.array(v)
