pragma solidity >=0.4.20 <=0.6.0;

import "@openzeppelin/contracts/access/roles/WhitelistedRole.sol";

contract SimpleResourceAllocation3 {

    // quotes of all resources, the index of the array represents the Id of that resource
    uint[] private quotes; 
    
    // resource ownership of alll users
    mapping(address => mapping(uint => bool)) internal ownership;

    constructor (uint[] memory initQuotes) public {
        quotes = initQuotes;
    }
    
    /// @dev request resource from system
    /// @param _resId  unsigned integer - the Id of the resource wanted
    /// @return _success bool - success status of this transaction
    function request(uint _resId) public  returns (bool _success) {
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
    
    /// @dev release resource to the system
    /// @param _resId  unsigned integer - the Id of the resource to be released
    /// @return _success bool - success status of this transaction
    function release(uint _resId) public returns (bool _success) {
        require(_resId >= 0 && _resId < quotes.length, "Invalid resource Id!");
        require(ownership[msg.sender][_resId] == true, "You dont't have this resource to release!");
        ownership[msg.sender][_resId] = false;
        quotes[_resId]++;
        _success = true;
    }
    
    /// @dev query the quote of all available resources 
    /// @return _allQuotes array - current quote of all resources
    function viewAllQuotes() external view returns (uint[] memory) {
        return quotes;
    }
    
    /// @dev query my resources 
    /// @return _myResources array - tell what resources I have had
    function viewMyResources() external view returns (bool[] memory) {
        bool[] memory temp = new bool[](quotes.length);
        for (uint i = 0; i < quotes.length; i++) {
            temp[i] = ownership[msg.sender][i];
        }
        return temp;
    }
    
}
