/* eslint-disable */
<template>
  <div>
    <el-row type="flex"  justify="center" align="middle">
      <el-col :span="16">
        <h1 style="text-align: center;">ResAloc Admin Page</h1>
        <el-input type="text" v-model="toAdd">
          <el-button @click="addWhitelisted" slot="append">Add Whitelisted</el-button>
        </el-input>
        <el-input type="text" v-model="toRemove">
          <el-button @click="removeWhitelisted" slot="append" type="danger">Del Whitelisted</el-button>
        </el-input>
        <el-input type="text" v-model="toAddAdmin">
          <el-button @click="addAdmin" slot="append">Add Admin</el-button>
        </el-input>
        <el-input type="text" v-model="toRemoveAdmin">
          <el-button @click="removeAdmin" slot="append" type="danger">Del Admin</el-button>
        </el-input>
        <el-input type="text" v-model="isWhite">
          <el-button @click="isWhiteListed" slot="append" type="danger">Is Whitelisted?</el-button>
        </el-input>
      </el-col>
    </el-row>
  </div>
</template>

<script>
import { resalocContract } from '../../utils/getContractAndAccount'

export default {
  props: ['account'],
  data () {
    return {
      toAdd: null,
      toRemove: null,
      toAddAdmin: null,
      toRemoveAdmin: null,
      isWhite: null
    }
  },
  methods: {
    addWhitelisted () {
      const that = this
      resalocContract.deployed().then(async function (instance) {
        instance.addWhitelisted(that.toAdd, {from: that.account})
      })
    },
    removeWhitelisted () {
      const that = this
      resalocContract.deployed().then(async function (instance) {
        instance.removeWhitelisted(that.toRemove, {from: that.account})
      })
    },
    addAdmin () {
      const that = this
      resalocContract.deployed().then(async function (instance) {
        instance.addWhitelistAdmin(that.toAddAdmin, {from: that.account})
      })
    },
    removeAdmin () {
      const that = this
      resalocContract.deployed().then(async function (instance) {
        instance.removeWhitelisted(that.toRemoveAdmin, {from: that.account})
      })
    },
    isWhiteListed () {
      const that = this
      resalocContract.deployed().then(async function (instance) {
        instance.isWhitelisted(that.isWhite).then(function (result) {
          alert(result)
        })
      })
    }
  }
}

</script>

<style>

</style>
