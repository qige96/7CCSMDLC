const { initialQuotes, initialWhilelisted } = require("../settings")

const ResAloc = artifacts.require("SimpleResourceAllocation3");

module.exports = function(deployer) {
  // deployment steps
  deployer.deploy(ResAloc, initialQuotes, initialWhilelisted);
};

