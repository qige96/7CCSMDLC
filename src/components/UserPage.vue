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
import { resalocContract } from '../../utils/getContractAndAccount'
import { ownershipLimit } from '../../settings'

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
  },
  methods: {
    refresh () {
      let that = this
      resalocContract.deployed().then(async function (instance) {
        const res = await instance.viewAllQuotes.call()
        for (let i in res) {
          res[i] = res[i].toNumber()
        }
        that.quotes = res
        // console.log(res)
      })
      resalocContract.deployed().then(async function (instance) {
        const res = await instance.viewMyResources.call()
        that.myResourcss = res
        // console.log(res)
      })
    },
    async request (reqId) {
      let that = this
      if (this.ownData.length >= ownershipLimit) {
        alert(`Sorry, you can only poccess no more than ${ownershipLimit} resources!`)
      } else {
        resalocContract.deployed().then(async function (instance) {
          instance.request(reqId, {from: that.account}).then(function (result) {
            that.refresh()
            // console.log(result)
          })
        })
      }
    },
    release (relId) {
      let that = this
      console.log('relId', relId)
      resalocContract.deployed().then(async function (instance) {
        instance.release(relId, {from: that.account}).then(function (val) {
          that.refresh()
          // console.log(val)
        })
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
