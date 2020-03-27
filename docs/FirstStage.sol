// file: FirstStage.sol 
pragma solidity >=0.4.20 <=0.6.0;

import "./utils.sol";
import "@openzeppelin/contracts/access/roles/WhitelistedRole.sol";

contract FirstStage is WhitelistedRole {

    // quotes of all resources, the index of the array represents the Id of that resource
    uint[] public quotes; 
    // resource ownership of alll users
    mapping(uint => mapping(address => bool)) internal resApplication;
    // resource resApplication limit for every users, 4 by default
    uint ownershipLimit = 4;
    // user accounts allowed to participate in this event
    address[] public allowedUsers;
    // result for initial allocation
    mapping(address => uint[]) public alloc;
    
    bool private allocationExecuted = false;
    mapping(uint => address[]) private tempAlloc;
    
    event successApply (address _requester, uint resId);
    event successWithdraw (address _releaser, uint resId);

    /// @dev initialise and deploy the contrat
    /// @param _limit - maximum resource holding for each user
    /// @param _initQuotes - initial reosurces quotes
    /// @param _whitelisted - accounts allowed to participate in this event
    constructor (uint _limit, uint[] memory _initQuotes, address[] memory _whitelisteds) public {
        ownershipLimit = _limit;
        quotes = _initQuotes;
        allowedUsers = _whitelisteds;
        for (uint i = 0; i < _whitelisteds.length; i++){
            addWhitelisted(_whitelisteds[i]);
        }
    }
    
    modifier notAllocated () {
        require(allocationExecuted == false, "Allocation has been executed!");
        _;
    }
    
    /// @dev request resource from system
    /// @param _resId  unsigned integer - the Id of the resource wanted
    /// @return _success bool - success status of this transaction
    function applyFor(uint _resId) public onlyWhitelisted notAllocated returns (bool _success) {
        require(_resId >= 0 && _resId < quotes.length, "Invalid resource Id!");
        require(resApplication[_resId][msg.sender] == false, "You have applied this resoource!");
        resApplication[_resId][msg.sender] = true;
        _success = true;
    }
    
    /// @dev release resource to the system
    /// @param _resId  unsigned integer - the Id of the resource to be released
    /// @return _success bool - success status of this transaction
    function withdraw(uint _resId) public onlyWhitelisted notAllocated returns (bool _success) {
        require(_resId >= 0 && _resId < quotes.length, "Invalid resource Id!");
        require(resApplication[_resId][msg.sender] == true, "You didn't apply this resoource!");
        resApplication[_resId][msg.sender] = false;
        _success = true;
    }
    
    /// @dev query my resources 
    /// @return _myResources array - tell what resources I have had
    function viewMyApplication() external view returns (bool[] memory) {
        bool[] memory temp = new bool[](quotes.length);
        for (uint i = 0; i < quotes.length; i++) {
            temp[i] = resApplication[i][msg.sender];
        }
        return temp;
    }
    
    /// @dev initial allocation
    function allocate() notAllocated public {
        for (uint i = 0; i < quotes.length; i++) {
            for (uint j = 0; j < allowedUsers.length; j++){
                if (resApplication[i][allowedUsers[j]]) {
                    tempAlloc[i].push(allowedUsers[j]);
                    // alloc[allowedUsers[j]].push(i);
                }
            }
        }
        for (uint i = 0; i < quotes.length; i++) {
            if (tempAlloc[i].length > quotes[i]) {
                // do nothing
            } else {
                for (uint j = 0; j < allowedUsers.length; j++){
                    if (resApplication[i][allowedUsers[j]]) {
                        alloc[allowedUsers[j]].push(i);
                    }
                }
            }
        }
        
        allocationExecuted = true;
    }

    /// @dev obtain result of initial allocation
    /// @return array - initial allocation result for a specific user
    function viewAllocationResult(address _user) view external returns (uint[] memory) {
        require(allocationExecuted == true, "Allocation not executed!");
        return alloc[_user];
    }
    
}
