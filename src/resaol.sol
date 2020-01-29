pragma solidity >=0.4.25;

contract SimpleResourceAllocation3 {

    uint[] public quotes;
    mapping(address => mapping(uint => bool)) internal ownership;
    
    constructor (uint[] memory initQuotes) public {
        quotes = initQuotes;
    }
    
    function request(uint res_id) public returns (bool success) {
        require(res_id >= 0 && res_id < quotes.length, "Invalid resource Id!");
        require(ownership[msg.sender][res_id] == false, "You have allocated this resource!");
        if (quotes[res_id] > 0) {
            ownership[msg.sender][res_id] = true;
            quotes[res_id] -= 1;
            success = true;
        } else {
            success = false;
        }
    }
    
    function release(uint res_id) public returns (bool success) {
        require(res_id >= 0 && res_id < quotes.length, "Invalid resource Id!");
        require(ownership[msg.sender][res_id] == true, "You dont't have this resource to release!");
        ownership[msg.sender][res_id] = false;
        quotes[res_id]++;
        success = true;
    }
    
    
    function viewAllQuotes() public view returns (uint[] memory allQuotes) {
        allQuotes = quotes;
    }
    
    function viewMyResources() public view returns (bool[] memory myResources) {
        bool[] memory temp = new bool[](quotes.length);
        for (uint i = 0; i < quotes.length; i++) {
            temp[i] = ownership[msg.sender][i];
        }
        myResources = temp;
    }
    
}
