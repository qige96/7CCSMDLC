/* eslint-disable */
<template>
  <div>
    <el-container>
      <el-header style="text-align: center;">
        <h1>Simple Decentralised Resource Allocation System</h1>
      </el-header>
      <el-container>
        <el-main>
          <el-row type="flex" justify="center" :gutter="20" >
            <el-col :span=10>
              <h3>My Resources</h3>
              <el-table :data="ownData">
                <el-table-column prop="resourceId" label="Resource Id">
                </el-table-column>
                <el-table-column label="Operation">
                  <template slot-scope="scope">
                  <el-button type="warning" plain size="small" @click="release(scope.row.resourceId)">release</el-button>
                  </template>
                </el-table-column>
              </el-table>
            </el-col>
            <el-col :span=10>
              <h3>Quotes</h3>
              <el-table :data="quoteData">
                <el-table-column prop="resourceId" label="Resource Id">
                </el-table-column>
                <el-table-column prop="quote" label="Quote">
                </el-table-column>
                <el-table-column label="Operation">
                  <template slot-scope="scope">
                  <el-button type="primary" plain size="small" @click="request(scope.row.resourceId)">request</el-button>
                  </template>
                </el-table-column>
              </el-table>
            </el-col>
          </el-row>
        </el-main>
      </el-container>
      <button @click="refresh">refresh</button>
      <el-footer class="footer">
        <p>7CCSMDLC Distributed ledgers and crypto-currencies(19~20 SEM2 000001)</p>
        <p>King's College London</p>
        <p>by BlockDance</p>
      </el-footer>
    </el-container>
  </div>
</template>

<script>
import { resalocContractInstance } from '../../utils/getContractAndAccount'

export default {
  props: ['account'],
  data () {
    return {
      quotes: [],
      myResourcss: [],
      reqId: null,
      relId: null,
      whitelisted: null
    }
  },
  computed: {
    ownData () {
      let resTable = []
      for (let i in this.myResourcss) {
        if (this.myResourcss[i]) {
          resTable.push({resourceId: i})
        }
      }
      return resTable
    },
    quoteData () {
      let resTable = []
      for (let i in this.quotes) {
        if (!this.myResourcss[i]) {
          resTable.push({resourceId: i, quote: this.quotes[i]})
        }
      }
      return resTable
    }
  },
  async mounted () {
    this.refresh()
    let that = this
    resalocContractInstance.events.successRequest(function (error, event) {
      console.log(error)
    })
      .on('data', (log) => {
        console.log(log)
        that.refresh()
      })
      .on('changed', (log) => { console.log(log) })
      .on('error', (log) => { console.log(log) })
    resalocContractInstance.events.successRelease(function (error, event) {
      console.log(error)
    })
      .on('data', (log) => {
        console.log(log)
        that.refresh()
      })
      .on('changed', (log) => { console.log(log) })
      .on('error', (log) => { console.log(log) })
  },
  methods: {
    refresh () {
      // console.log(this.account)
      resalocContractInstance.methods.viewAllQuotes().call(
        {gas: 300000, from: this.account},
        (err, result) => {
          if (err) {
            console.log('err', err)
          } else {
            this.quotes = result
          }
        })

      resalocContractInstance.methods.viewMyResources().call(
        {gas: 300000, from: this.account},
        (err, result) => {
          if (err) {
            console.log('err', err)
          } else {
            this.myResourcss = result
          }
        })
    },
    request (reqId) {
      console.log('reqId', reqId)
      resalocContractInstance.methods.request(reqId).send({from: this.account})
        .on('transactionHash', function (TxHash) {
          console.log(TxHash)
        })
        .on('confirmation', function (confirmationNumber, receipt) {
          console.log(receipt)
        })
        .on('error', function (err) {
          console.log(err)
        })
    },
    release (relId) {
      console.log('relId', relId)
      resalocContractInstance.methods.release(relId).send({from: this.account})
        .on('transactionHash', function (TxHash) {
          console.log(TxHash)
        })
        .on('confirmation', function (confirmationNumber, receipt) {
          console.log(receipt)
        })
        .on('error', function (err) {
          console.log(err)
        })
    }
  }
}

</script>

<style>
  body {
    display: flex;
    flex-direction: column;
  }
  .el-row {
    height: 100%;
  }
  .el-footer {
    text-align: center;
    font-size: x-small;
  }
</style>
