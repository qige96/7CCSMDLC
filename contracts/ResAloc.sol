pragma solidity >=0.4.20 <=0.6.0;

import "@openzeppelin/contracts/access/roles/WhitelistedRole.sol";

contract SimpleResourceAllocation3 is WhitelistedRole {

    // quotes of all resources, the index of the array represents the Id of that resource
    uint[] private quotes; 
    
    // resource ownership of alll users
    mapping(address => mapping(uint => bool)) internal ownership;
    
    event successRequest(address _requester, uint resId);
    event successRelease(address _releaser, uint resId);

    constructor (uint[] memory _initQuotes, address[] memory _whitelisteds) public {
        quotes = _initQuotes;
        for (uint i = 0; i < _whitelisteds.length; i++){
            addWhitelisted(_whitelisteds[i]);
        }
    }
    
    /// @dev request resource from system
    /// @param _resId  unsigned integer - the Id of the resource wanted
    /// @return _success bool - success status of this transaction
    function request(uint _resId) public onlyWhitelisted returns (bool _success) {
        require(_resId >= 0 && _resId < quotes.length, "Invalid resource Id!");
        require(ownership[msg.sender][_resId] == false, "You have allocated this resource!");
        if (quotes[_resId] > 0) {
            ownership[msg.sender][_resId] = true;
            quotes[_resId] -= 1;
            emit successRequest(msg.sender, _resId);
            _success = true;
        } else {
            _success = false;
        }
    }
    
    /// @dev release resource to the system
    /// @param _resId  unsigned integer - the Id of the resource to be released
    /// @return _success bool - success status of this transaction
    function release(uint _resId) public onlyWhitelisted returns (bool _success) {
        require(_resId >= 0 && _resId < quotes.length, "Invalid resource Id!");
        require(ownership[msg.sender][_resId] == true, "You dont't have this resource to release!");
        ownership[msg.sender][_resId] = false;
        quotes[_resId]++;
        emit successRelease(msg.sender, _resId);
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