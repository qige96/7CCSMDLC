pragma solidity >=0.4.20 <=0.6.0;

library utils {
    
    function bytes32ToString (bytes32 data) public pure returns (string memory) {
        bytes memory bytesString = new bytes(32);
        for (uint j=0; j<32; j++) {
            byte char = byte(bytes32(uint(data) * 2 ** (8 * j)));
            if (char != 0) {
                bytesString[j] = char;
            }
        }
        return string(bytesString);
    }
    
    function toBytes(uint256 x) public pure returns (bytes memory b) {
        b = new bytes(32);
        assembly { mstore(add(b, 32), x) }
    }

    function randint (uint _min, uint _max) public view returns (uint) {
        bytes memory rand = toBytes(now);
        return uint(keccak256(rand)) % (_max - _min) + _min;
    }
}
