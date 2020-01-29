pragma solidity >=0.4.25;

contract SimpleResourceAllocation2 {

    uint[] public quotes;
    mapping(address => mapping(uint => uint)) internal ownership;
    
    constructor (uint[] memory initQuotes) public {
        quotes = initQuotes;
    }
    
    function request(uint res_id) public returns (bool success) {
        require(res_id >= 0 && res_id < quotes.length, "Invalid resource Id!");
        if (quotes[res_id] > 0) {
            ownership[msg.sender][res_id] += 1;
            quotes[res_id] -= 1;
            success = true;
        } else {
            success = false;
        }
    }
    
    function release(uint res_id, uint amount) public returns (bool success) {
        require(res_id >= 0 && res_id < quotes.length, "Invalid resource Id!");
        require(ownership[msg.sender][res_id] >= amount, "You dont't have this resource to release!");
        ownership[msg.sender][res_id] -= amount;
        quotes[res_id] += amount;
        success = true;
    }
    
    function viewResourceAmount(uint res_id) public view returns (uint resAmount) {
        resAmount = ownership[msg.sender][res_id];
    }
    
    function viewAllQuotes() public view returns (uint[] memory allQuotes) {
        allQuotes = quotes;
    }
    
    function viewAllAmount() public view returns (uint[] memory allAmount) {
        uint[] memory temp = new uint[](quotes.length);
        for (uint i = 0; i < quotes.length; i++) {
            temp[i] = ownership[msg.sender][i];
        }
        allAmount = temp;
    }
    
}
