pragma solidity >=0.4.25;

contract SimpleResourceAllocation3 {

    uint[] public quotes;
    mapping(address => mapping(uint => bool)) internal ownership;
    
    constructor (uint[] memory initQuotes) public {
        quotes = initQuotes;
    }
    
    function request(uint _resId) public returns (bool _success) {
        require(_resId >= 0 && _resId < quotes.length, "Invalid resource Id!");
        require(ownership[msg.sender][_resId] == false, "You have allocated this resource!");
        if (quotes[_resId] > 0) {
            ownership[msg.sender][_resId] = true;
            quotes[_resId] -= 1;
            _success = true;
        } else {
            _success = false;
        }
    }
    
    function release(uint _resId) public returns (bool _success) {
        require(_resId >= 0 && _resId < quotes.length, "Invalid resource Id!");
        require(ownership[msg.sender][_resId] == true, "You dont't have this resource to release!");
        ownership[msg.sender][_resId] = false;
        quotes[_resId]++;
        _success = true;
    }
    
    
    function viewAllQuotes() public view returns (uint[] memory _allQuotes) {
        _allQuotes = quotes;
    }
    
    function viewMyResources() public view returns (bool[] memory _myResources) {
        bool[] memory temp = new bool[](quotes.length);
        for (uint i = 0; i < quotes.length; i++) {
            temp[i] = ownership[msg.sender][i];
        }
        _myResources = temp;
    }
    
}
